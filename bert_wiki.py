## 
## Referencia:
## https://colab.research.google.com/drive/1uSlWtJdZmLrI3FCNIlUHFxwAJiSu2J0-

print('19/01/2024')
## bibliotecas a serem incluidas nos SIF
#transformers
#sentence-transformers


# dataset wiki
import gzip
import json

#retriever e re-ranker
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
import os
import csv
import pickle
import time

#reader
import torch
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering


# segundo teste com dataset de passagens da wikipedia
# We use the Bi-Encoder to encode all passages, so that we can use it with semantic search



#--------------- funcoes (início) ------------------
def load_dataset(max_corpus_size):
    # Some local file to cache computed embeddings
    embedding_cache_path = "data/wikipedia/nq-distilbert-base-v1-{}-size-{}.pkl".format(model_name.replace("/", "_"), max_corpus_size)
    # Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):
        print(f'Embeddings cache {embedding_cache_path} nao encontrados localmente...')

        # Check if the dataset exists. If not, download and extract
        # Download dataset if needed
        if not os.path.exists(dataset_path):
            print(f'Dataset {dataset_path} nao encontrado local...')
            util.http_get(url, dataset_path)

        passages = []
        print(f'Lendo dataset {dataset_path}...')
        with gzip.open(dataset_path, "rt", encoding="utf8") as fIn:
            for line in fIn:
                data = json.loads(line.strip())
                for paragraph in data["paragraphs"]:
                # We encode the passages as [title, text]
                    passages.append([data["title"],paragraph])   ## varias passages em cada paragraph
            if (max_corpus_size != 0):
                passages = passages[0:max_corpus_size]
            print("Passages:", len(passages))
        
        print(f'Fazendo embbeding do dataset')
        corpus_embeddings = retriever_biencoder_model.encode(passages, convert_to_tensor=True, show_progress_bar=True)


        #corpus_embeddings = torch.load(embedding_cache_path, map_location=torch.device('cpu'))  ## testar com GPU
        
        print(f'Salvando localmente os embeddings em {embedding_cache_path} ')
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({"passages": passages, "embeddings": corpus_embeddings}, fOut)
    else:
        print(f'Embeddings cache {embedding_cache_path} encontrado localmente...')
        print("Aguarde carregando....")

        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            passages = cache_data["passages"][0:max_corpus_size]
            corpus_embeddings = cache_data["embeddings"][0:max_corpus_size]
            corpus_embeddings = corpus_embeddings.float()  # Convert embedding file to float

        ##corpus_embeddings = torch.load(embeddings_filepath, map_location=torch.device('cpu'))  ## testar com GPU
        #This way the model will be loaded on the CPU device, even if a CUDA device was used to train it.
        ## corpus_embeddings = corpus_embeddings.float() 
    ###############################
    print("")
    print("Corpus loaded with {} passages / embeddings".format(len(passages)))

    return corpus_embeddings,passages

def cosine_similarity(inp_question,corpus_embeddings,num_candidates):

    # First, retrieve candidates using cosine similarity search
    start_time = time.time()
    question_embedding = retriever_biencoder_model.encode(inp_question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=num_candidates)
    hits = hits[0]  # Get the hits for the first query

    print(f'Cosine-Similarity para top-{num_candidates} levou {(time.time() - start_time)} seconds')
    print(f'imprimindo Top 5 dos {num_candidates} hits with cosine-similarity:')
    for hit in hits[0:5]:
        print("\t{:.3f}\t{}".format(hit["score"], corpus_sentences[hit["corpus_id"]]))
    print('---------------------------------------------------------------------------')
    #hits = hits[0:5]  ## forcando somente top5
    return hits


def re_ranking(inp_question,corpus_sentences,hits,num_candidates):

    # Now, do the re-ranking with the cross-encoder
    start_time = time.time()
    sentence_pairs = [[inp_question, corpus_sentences[hit["corpus_id"]][1]] for hit in hits]  # montou os i pares pergunta:hit[i]
    ### atencao para o [1], pega a passage, pois o corpus_sentences[hit["corpus_id"]] passou a ser uma lista!! 
    ce_scores = rerank_cross_encoder_model.predict(sentence_pairs)

    for idx in range(len(hits)):
        hits[idx]["cross-encoder_score"] = ce_scores[idx]

    # Sort list by CrossEncoder scores
    hits = sorted(hits, key=lambda x: x["cross-encoder_score"], reverse=True)
    print("\nRe-ranking with CrossEncoder took {:.3f} seconds".format(time.time() - start_time))
    print(f'Imprimindo Top 5 dos {num_candidates} hits with CrossEncoder:')

    sentences = []

    for hit in hits[0:5]:
        print("\t{:.3f}\t{}".format(hit["cross-encoder_score"], corpus_sentences[hit["corpus_id"]]))
        print(f'Corpus id: {hit["corpus_id"]}')
        #print(f'passages[]')
        sentences.append(corpus_sentences[hit["corpus_id"]])
    
    return(hits)

def get_answer(input_ids,segment_ids):
        # Run our example through the model.

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
#--------------- funcoes (fim) ------------------

debug_mode = False
max_corpus_size = 1000  # 0 significa sem restricao
dataset_path = "data/wikipedia/simplewiki-2020-11-01.jsonl.gz"

if __name__ == '__main__':
    if debug_mode == True:
        import debugpy
        debugpy.listen(7011)
        print("Waiting for debugger attach")
        debugpy.wait_for_client() 
    print('hello 2')
    i = 1
    print('hello 3')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"using device: {torch.cuda.get_device_name()}")
    print(device)

    # retriever
    model_name = "nq-distilbert-base-v1"
    retriever_biencoder_model = SentenceTransformer(model_name)
    top_k = 5  # Number of passages we want to retrieve with the bi-encoder

    ## reranker
    rerank_cross_encoder_model = CrossEncoder("cross-encoder/stsb-roberta-base")
    #The model will predict scores for the pairs ('Sentence 1', 'Sentence 2')

    # Reader: carrega o modelo de Q&A
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    reader_model_qa = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


    #dataset contendo texto wiki
    url = "http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz"
    dataset_path = "data/wikipedia/simplewiki-2020-11-01.jsonl.gz"


    # 0 -carrega os embeddings do cache ou faz o download do dataset e calcula os embedding
    corpus_embeddings,corpus_sentences = load_dataset(max_corpus_size)  ## 20.000  perguntas
    print(f'dimensao do vetor: {len(corpus_embeddings[0])}')
    print(f'dimensao da passage 0: {len(corpus_sentences[0][1])}')


while True:
    inp_question = input("Entre com a question: ")
    num_candidates = int(input("Entre com numero para Top-K candidatos: "))

    ## 01 - similaridade da question com os embedding
    hits = cosine_similarity(inp_question,corpus_embeddings,num_candidates)

    ## 02 - re ranking com um cross-encoder
    hits = re_ranking(inp_question,corpus_sentences,hits,num_candidates)

    ## 03 - tokeniza a [pergunta:documento]
    document = corpus_sentences[hits[0]["corpus_id"]][1]   ## atencao com o [1]
    # search feito somente no primeiro documento
    input_ids, segment_ids, tokens = get_token_from_pair(inp_question,document)   

    ## 04 - obtem a resposta
    answer = get_answer(input_ids,segment_ids)  # inputs_ids = tokens representing our input text = [pergunta:resposta]
                                                # segment IDs = 1/0 to differentiate question from answer_text

    print(f'question    :  {inp_question}')
    print(f'documento id: {hits[0]["corpus_id"]}')
    #print(f'documento title: {corpus_sentences[hits[0]["corpus_id"]]}')
    print(f'documento: {corpus_sentences[hits[0]["corpus_id"]]}')

    print(f'Answer:    {answer}')
    print('')