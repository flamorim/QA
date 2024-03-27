
## who rider astana team?
### which is ISBN from Frasier?
## who published ISBN 0-8487-1550-0?                              none agg
## who many times dir Quick-Step team win a ride?                     count aggregation
## what average time did team Barloworld win the ride?     average aggregation
## who won from astana team?                                                NONE aggregation > Enyu Valchev
## how much cost a house?                                         
## who riders from quick_step team?
## which riders are from ITA?  COUNT
## Referencia:
## https://colab.research.google.com/drive/1uSlWtJdZmLrI3FCNIlUHFxwAJiSu2J0-   artigo dos span's
## https://huggingface.co/docs/transformers/model_doc/tapas#usage-inference
## https://github.com/pinecone-io/examples/blob/master/learn/search/question-answering/table-qa.ipynb

## dataset: Question Answering (OTT-QA) dataset
## retriever: SentenceTransformer FT com tabelas ("deepset/all-mpnet-base-v2-table", device=device)
## reranking: cross-encoder/stsb-roberta-base
## reader:    Tapas

print('19/01/2024')

import gzip
import json
import os
import warnings
#warnings.filterwarnings("ignore")

#retriever 
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import pandas as pd
 
# e re-ranker
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
import os
import csv
import pickle
import time

#reader
import torch
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering

#--------------- funcoes (início) ------------------
def get_dataset(max_corpus_size):
    # Some local file to cache computed embeddings
    embedding_cache_path = "/home/Bert/data/ott-qa/{}-size-{}.pkl".format(retriever_model_name.replace("/", "_"), max_corpus_size)
    # Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):
        print(f'Embeddings encontrado em cache {embedding_cache_path}')
        if not os.path.exists(dataset_path):
            print(f'Dataset {dataset_path} nao encontrado local...')
            corpus_table = load_dataset(url, split="train")  # baixando dataset
            tables = create_tables_list(corpus_table)   # lista de df
            if max_corpus_size != 0:
                tables = tables[0:max_corpus_size]
            
        #from tqdm.auto import tqdm
        processed_tables = _preprocess_tables(tables)   ## convert para csv

        corpus_embeddings = retriever_biencoder_model.encode(processed_tables, convert_to_tensor=True)
        print(f'Salvando localmente os embeddings e tabelas em {embedding_cache_path} ')
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({"tables": tables, "embeddings": corpus_embeddings}, fOut)
    else:
        print(f'Embeddings cache {embedding_cache_path} encontrado localmente...')
        print("Aguarde carregando....")

        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            tables = cache_data["tables"]
            corpus_embeddings = cache_data["embeddings"]
        if max_corpus_size != 0:
            tables = tables[0:max_corpus_size]
            corpus_embeddings = corpus_embeddings[0:max_corpus_size]
    print("")
    print("Corpus loaded with {} tables / embeddings".format(len(tables)))

    return corpus_embeddings,tables


def cosine_similarity(inp_question,corpus_embeddings,num_candidates):

    start_time = time.time()
    question_embedding = retriever_biencoder_model.encode(inp_question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=num_candidates)
    hits = hits[0]  # Get the hits for the first query

    print(f'Cosine-Similarity para top-{num_candidates} levou {(time.time() - start_time)} seconds')
    print(f'imprimindo Top 5 dos {num_candidates} hits with cosine-similarity:')
    for hit in hits[0:5]:
        print("\t{:.3f}\t{}".format(hit["score"], corpus_tables[hit["corpus_id"]]))
    print('---------------------------------------------------------------------------')

    return hits


def re_ranking(inp_question,passages_topk,hits):
    # Now, do the re-ranking with the cross-encoder
    start_time = time.time()
    sentence_pairs = [[inp_question, header_tabela] for header_tabela in passages_topk]  # montou os i pares pergunta:hit[i]
    cross_encoder_scores = rerank_cross_encoder_model.predict(sentence_pairs)

    for idx in range(len(hits)):
        hits[idx]["cross-encoder_score"] = cross_encoder_scores[idx]

    # Sort list by CrossEncoder scores
    hits = sorted(hits, key=lambda x: x["cross-encoder_score"], reverse=True)
    print("\nRe-ranking with CrossEncoder took {:.3f} seconds".format(time.time() - start_time))
    

 #   for hit in hits[0:5]:
 #       print("\t{:.3f}\t{}".format(hit["cross-encoder_score"], corpus_tables[hit["corpus_id"]]))
 #       print(f'Corpus id: {hit["corpus_id"]}')
    
    return(hits)

def get_answer(input_ids,segment_ids):

    start_time = time.time()
    outputs = reader_model_qa(torch.tensor([input_ids]), # The tokens representing our input text.
                                token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                                return_dict=True)
    
    print("\nReader with CrossEncoder took {:.3f} seconds".format(time.time() - start_time))

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    return answer 

    #if (answer_start >= answer_end):
    #    return ('error')
        # Combine the tokens in the answer and print it out.
    #answer = ' '.join(tokens[answer_start:answer_end+1])

    
    #print('Answer: "' + answer + '"')

    #return answer



def get_token_from_pair(question,document):
       #CONCATENANDO, COMO SERA FEITO NA LISTA DAS TOP-K
        # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, document)
    print('The input has a total of {:} tokens.'.format(len(input_ids)))
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # For each token and its id...
    #for token, id in zip(tokens, input_ids):
    #    # If this is the [SEP] token, add some space around it to make it stand out.
    #    if id == tokenizer.sep_token_id:
    #        print('')
    #    # Print the token string and its ID in two columns.
    #    print('{:<12} {:>6,}'.format(token, id))
    #    if id == tokenizer.sep_token_id:
    #        print('')
    ##########

        # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

        # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

        # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

        # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

        # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    return (input_ids,segment_ids,tokens)



def create_tables_list(corpus_table):
    # store all tables in the tables list
    tables = []
    # loop through the dataset and convert tabular data to pandas dataframes
    for doc in corpus_table:
        table = pd.DataFrame(doc["data"], columns=doc["header"])
        tables.append(table)
    return tables

def _preprocess_tables(tables: list):
    processed = []
    # loop through all tables
    for table in tables:
        # convert the table to csv and
        processed_table = "\n".join([table.to_csv(index=False)])
        # add the processed table to processed list
        processed.append(processed_table)
    return processed

#--------------- funcoes (fim) ------------------

debug_mode = False #True
max_corpus_size = 0 #25  # 0 significa sem restricao
#dataset_path = "data/wikipedia/simplewiki-2020-11-01.jsonl.gz"

if __name__ == '__main__':
    if debug_mode == True:
        import debugpy
        debugpy.listen(7110)
        print("Waiting for debugger attach")
        debugpy.wait_for_client() 
    print('hello 2')
    i = 1
    print('hello 3')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # retriever
    retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
    retriever_biencoder_model = SentenceTransformer(retriever_model_name, device=device)
    top_k = 5  # Number of passages we want to retrieve with the bi-encoder

    ## reranker
    rerank_cross_encoder_model = CrossEncoder("cross-encoder/stsb-roberta-base")
    #The model will predict scores for the pairs ('Sentence 1', 'Sentence 2')

    #1) FUNCIONA
    # Reader: carrega o modelo de Q&A weak supervised
    #reader_model_name = "google/tapas-large-finetuned-wtq"
    #reader_model = TapasForQuestionAnswering.from_pretrained(reader_model_name)
    # load the tokenizer and the model from huggingface model hub
    #reader_tokenizer = TapasTokenizer.from_pretrained(reader_model_name)

    #2) https://metatext.io/models/google-tapas-large-finetuned-wikisql-supervised
    # or, the base sized model with WikiSQL configuration
    # strong supervised
    #from transformers import AutoModel, AutoTokenizer 
    #reader_model_name  = "google-base-finetuned-wikisql-supervised"
    #reader_model = AutoModel.from_pretrained(reader_model_name)
    #reader_tokenizer = AutoTokenizer.from_pretrained(reader_model_name)

    #3)https://huggingface.co/google/tapas-large-finetuned-wikisql-supervised
    #config = TapasConfig("google-base-finetuned-wikisql-supervised")
    #reader_model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

    #4) https://www.modelbit.com/blog/deploying-googles-table-q-a-model-tapas-to-a-rest-api
    from transformers import TapasConfig
    reader_model_name = "google/tapas-large-finetuned-wikisql-supervised"
    config = TapasConfig(reader_model_name)
    reader_model = TapasForQuestionAnswering.from_pretrained(reader_model_name)
    reader_tokenizer = TapasTokenizer.from_pretrained(reader_model_name)


    #dataset contendo tabela
    url = "ashraq/ott-qa-20k"
    dataset_path = "/home/Bert/data/ott-qa/ott-qa-20k"


    # 0 -carrega os embeddings do cache ou faz o download do dataset e calcula os embedding
    corpus_embeddings,corpus_tables = get_dataset(max_corpus_size)  ## 20.000  perguntas
    print(f'dimensao do vetor: {len(corpus_embeddings[0])}')



while True:
    inp_question = input("Entre com a question: ")
    num_candidates = int(input("Entre com numero para Top-K candidatos: "))

    ## 01 - similaridade da question com os embedding
    hits = cosine_similarity(inp_question,corpus_embeddings,num_candidates)

    ## 02 - re ranking com um cross-encoder | baseado nas passages x question
    passages_topk = []

    for idx in range(top_k): # TRANSFORMAR EM DICIONARIO ID,PASSAGE
        # Obter os nomes das colunas como uma lista
        table_headers = corpus_tables[hits[idx]["corpus_id"]].columns.tolist()
        string_table_headers = ', '.join(table_headers)
        passages_topk.append(string_table_headers)

    hits_reranked = re_ranking(inp_question,passages_topk,hits)

    print('Posicao  |  antes   |    atual')
    for idx in range(min(5, top_k)):
        print(f'     {idx}   |   {hits[idx]["corpus_id"]}   |     {hits_reranked[idx]["corpus_id"]} ')
    print('---------------------------------------------------------------------------')

    ## 03 - tokeniza a [pergunta:documento]
    queries = []
    queries.append(inp_question)  # pegando apenas 1 pergunta
    #####################
    table_header = corpus_tables[hits_reranked[0]["corpus_id"]].columns.tolist()
    passage = ', '.join(table_header)
    
    entrada = corpus_tables[hits_reranked[0]["corpus_id"]]
    entrada = f"{passage}\n{entrada}"
    
    #####################    
    #inputs = reader_tokenizer(table=entrada, queries=queries, padding="max_length", return_tensors="pt")
    inputs = reader_tokenizer(table=corpus_tables[hits_reranked[0]["corpus_id"]], queries=queries, padding="max_length", return_tensors="pt")
    outputs = reader_model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = reader_tokenizer.convert_logits_to_predictions(
    inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())

    id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            print("Resposta obtida em only a single cell:")
            answers.append(corpus_tables[hits_reranked[0]["corpus_id"]].iat[coordinates[0]])
        else:
            print("Resposta obtida em multiple cells")
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(corpus_tables[hits_reranked[0]["corpus_id"]].iat[coordinate])
            answers.append(", ".join(cell_values))

    print("")
    for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
        print(query)
        #if predicted_agg == "NONE":
        #    print("Predicted answer: " + answer)
        #else:
        print("Predicted answer: " + predicted_agg + " aggregation > " + answer)
    print('')
    print(f'Tabela id: {hits_reranked[0]["corpus_id"]}')
    print(f'{corpus_tables[hits_reranked[0]["corpus_id"]]}')
    print('')
    print('')
    print(f'Tabela id: {hits[0]["corpus_id"]}')
    print(f'{corpus_tables[hits[0]["corpus_id"]]}')
    print('')
    print('')
    print('---------------------------------------')
    