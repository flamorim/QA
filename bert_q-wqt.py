## ver referencia em 
##https://metatext.io/models/google-tapas-large-finetuned-wtq

print('hello')


if __name__ == '__main__':
    if True == True:
        import debugpy
        debugpy.listen(7200)
        print("Waiting for debugger attach")
        debugpy.wait_for_client() 

    print('hello 2')
    i = 1
    print('hello 3')
    #exit()
## bibliotecas a serem incluidas nos SIF
#pip install sentence-transformers
#pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+$CUDA121.html
  #965  python3
  #966  pip
  #967  pip show
  #968  pip show transformers
  #969  pip install transformers
  #970  python3 -m pip install --upgrade pip



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

# Modelo Q&A
#!pip install datasets sentence_transformers torch-scatter
'from datasets import load_dataset
# Modelo QA Retriever
import torch
from sentence_transformers import SentenceTransformer

#####
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"using device: {torch.cuda.get_device_name()}")
#exit()


teste = 'terceiro'


if teste == 'primeiro':
    # primeiro teste com dataset de perguntas
    # We use a BiEncoder (SentenceTransformer) that produces embeddings for questions.
    # We then search for similar questions using cosine similarity and identify the top 100 most similar questions
    model_name = "all-MiniLM-L6-v2"
    BiEncoder_model = SentenceTransformer(model_name)
    num_candidates = 500


    # primeiro dataset contendo perguntas - def load_dataset_squad(max_corpus_size):
    # Dataset we want to use
    url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
    dataset_path = "quora_duplicate_questions.tsv"
    max_corpus_size = 20

    # usa def load_dataset(max_corpus_size):

    #corpus_sentences = []
    #corpus_embeddings = []



if teste == 'segundo':
    # segundo teste com dataset de passagens da wikipedia - SQUAD
    # We use the Bi-Encoder to encode all passages, so that we can use it with semantic search
    model_name = "nq-distilbert-base-v1"
    BiEncoder_model = SentenceTransformer(model_name)
    top_k = 5  # Number of passages we want to retrieve with the bi-encoder
    # segundo dataset contendo sentencas do wikipedia
    dataset_path = "data/simplewiki-2020-11-01.jsonl.gz"
    #usa def load_dataset_wiki(max_corpus_size):



if teste == 'tttterceiro':
    #https://metatext.io/models/google-tapas-large-finetuned-wtq
    # Import generic wrappers
'   from transformers import AutoModel, AutoTokenizer
    
    # terceiro teste com modelo/dataset tabela WQT
    # Define the model repo
'   model_name = "google/tapas-large-finetuned-wtq" 
    # Download pytorch model
'   BiEncoder_model = AutoModel.from_pretrained(model_name)
'   tokenizer = AutoTokenizer.from_pretrained(model_name)


    # Transform input tokens 
    #inputs = tokenizer("Hello world!", return_tensors="pt")

    # Model apply
    #outputs = BiEncoder_model(**inputs)
    #exit()





# To refine the results, we use a CrossEncoder. A CrossEncoder gets both inputs (input_question, retrieved_question)
# and outputs a score 0...1 indicating the similarity.
cross_encoder_model = CrossEncoder("cross-encoder/stsb-roberta-base")
#The model will predict scores for the pairs ('Sentence 1', 'Sentence 2')

# carrega o modelo de Q&A
'tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
'model_qa = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')



def load_dataset_WTQ(max_corpus_size):
    #retorna o dataset e os embedding
    #load the dataset from huggingface datasets hub
    huggingface_hub = "ashraq/ott-qa-20k"
    data = load_dataset(huggingface_hub, split="train")
    if max_corpus_size != 0:
        data = data[0:max_corpus_size]
    return(data)


def load_dataset_wiki(max_corpus_size):

    # We use the Bi-Encoder to encode all passages, so that we can use it with semantic search
#    model_name = "nq-distilbert-base-v1"
#    bi_encoder = SentenceTransformer(model_name)
#    top_k = 5  # Number of passages we want to retrieve with the bi-encoder

    # As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
    # about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

    wikipedia_filepath = "data/simplewiki-2020-11-01.jsonl.gz"

    if not os.path.exists(wikipedia_filepath):  
        # baixando o datasedt
        util.http_get("http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz", wikipedia_filepath)

    passages = []
    #count_paragraphs = 0
    #count_passages = 0
    with gzip.open(wikipedia_filepath, "rt", encoding="utf8") as fIn:
        for line in fIn:
            data = json.loads(line.strip())
            for paragraph in data["paragraphs"]:
                # We encode the passages as [title, text]
                #passages.append([data["title"] + ' ' + paragraph])   ## varias passages em cada paragraph
                passages.append([data["title"],paragraph])   ## varias passages em cada paragraph
                #count_passages +=1                             ## ATENCAO, NAO E' MAIS TITULO,PARAGRAFO, MAS TITULO+PARAGRFO
                #if count_paragraphs == max_corpus_size:
                #    break
            #count_paragraphs +=1
    # If you like, you can also limit the number of passages you want to use
    #passages = passages[0:max_corpus_size]
    print("Passages:", len(passages))
    #print("Paragraphs:", count_paragraphs)
    #print("limite passages:", max_corpus_size)

    # To speed things up, pre-computed embeddings are downloaded.
    # The provided file encoded the passages with the model 'nq-distilbert-base-v1'
    if model_name == "nq-distilbert-base-v1":
        embeddings_filepath = "simplewiki-2020-11-01-nq-distilbert-base-v1.pt"
        if not os.path.exists(embeddings_filepath):
            util.http_get("http://sbert.net/datasets/simplewiki-2020-11-01-nq-distilbert-base-v1.pt", embeddings_filepath)

        corpus_embeddings = torch.load(embeddings_filepath, map_location=torch.device('cpu'))  ## testar com GPU
        #This way the model will be loaded on the CPU device, even if a CUDA device was used to train it.

        corpus_embeddings = corpus_embeddings.float()  # Convert embedding file to float
        #if torch.cuda.is_available():  retirado da GPU
            #corpus_embeddings = corpus_embeddings.to("cuda")
    else:  # Here, we compute the corpus_embeddings from scratch (which can take a while depending on the GPU)
        corpus_embeddings = BiEncoder_model.encode(passages, convert_to_tensor=True, show_progress_bar=True)

    return corpus_embeddings,passages

def load_dataset_squad(max_corpus_size):
    # Some local file to cache computed embeddings
    embedding_cache_path = "quora-embeddings-{}-size-{}.pkl".format(model_name.replace("/", "_"), max_corpus_size)

    print(f'cache path: {embedding_cache_path}')
    # Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):

        print("Fazendo o download do dataset")

        # Check if the dataset exists. If not, download and extract
        # Download dataset if needed
        if not os.path.exists(dataset_path):
            print("Download dataset")
            util.http_get(url, dataset_path)

        # Get all unique sentences from the file
        corpus_sentences = set()
        with open(dataset_path, encoding="utf8") as fIn:
            reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                corpus_sentences.add(row["question1"])
                if len(corpus_sentences) >= max_corpus_size:
                    break

                corpus_sentences.add(row["question2"])
                if len(corpus_sentences) >= max_corpus_size:
                    break

        corpus_sentences = list(corpus_sentences)
        print("Encode the corpus. This might take a while")
        corpus_embeddings = BiEncoder_model.encode(corpus_sentences, show_progress_bar=True, convert_to_tensor=True)

        print("Store file on disc")
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({"sentences": corpus_sentences, "embeddings": corpus_embeddings}, fOut)
    else:
        print("Carregando do cache os embedding do corpus previamente feitos")
        print("Aguarde....")

        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            corpus_sentences = cache_data["sentences"][0:max_corpus_size]
            corpus_embeddings = cache_data["embeddings"][0:max_corpus_size]

    ###############################
    print("")
    print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))

    return corpus_embeddings,corpus_sentences

def cosine_similarity(inp_question,corpus_embeddings,num_candidates):

    # First, retrieve candidates using cosine similarity search
    start_time = time.time()
    question_embedding = BiEncoder_model.encode(inp_question, convert_to_tensor=True)
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
    ce_scores = cross_encoder_model.predict(sentence_pairs)

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

    outputs = model_qa(torch.tensor([input_ids]), # The tokens representing our input text.
                                token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                                return_dict=True)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

        # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    if (answer_start >= answer_end):
        return ('error')

        # Combine the tokens in the answer and print it out.
    answer = ' '.join(tokens[answer_start:answer_end+1])

    print("\nReader with CrossEncoder took {:.3f} seconds".format(time.time() - start_time))

    #print('Answer: "' + answer + '"')

    return answer



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


################################################################
########################## MAIN ################################

# 0 -carrega os embeddings do cache ou faz o download do dataset e calcula os embedding
# primeiro teste com dataset de perguntas

if teste == 'primeiro':
    corpus_embeddings,corpus_sentences = load_dataset_squad(max_corpus_size)  ## 20.000  perguntas
    print(f'dimensao do vetor: {len(corpus_embeddings[0])}')
    print(f'dimensao da passage 0: {len(corpus_sentences[0])}')

if teste == 'segundo':
    corpus_embeddings,corpus_sentences = load_dataset_wiki(max_corpus_size)  ## 20.000  perguntas
    print(f'dimensao do vetor: {len(corpus_embeddings[0])}')
    print(f'dimensao da passage 0: {len(corpus_sentences[0])}')

import pandas as pd

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


if teste == 'terceiro':
    corpus_table = load_dataset_WTQ(0)
    tables = create_tables_list(corpus_table)
    print(tables[2])    # set device to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # load the table embedding model from huggingface models hub
    'NAO - BiEncoder_model = SentenceTransformer("deepset/all-mpnet-base-v2-table", device=device)   # retriever
    print(BiEncoder_model)

    # format all the dataframes in the tables list
    processed_tables = _preprocess_tables(tables[0:10])
    # display the formatted table
    processed_tables[2]

    from tqdm.auto import tqdm

    # we will use batches of 64
    batch_size = 64
    vectors = []
    for i in tqdm(range(0, len(processed_tables), batch_size)):
        # find end of batch
        i_end = min(i+batch_size, len(processed_tables))
        # extract batch
        batch = processed_tables[i:i_end]
        # generate embeddings for batch
        emb = BiEncoder_model.encode(batch).tolist()
        # create unique IDs ranging from zero to the total number of tables in the dataset
        ids = [f"{idx}" for idx in range(i, i_end)]
        # add all to upsert list
        to_upsert = list(zip(ids, emb))
        print(to_upsert[0][0])
        print(len(to_upsert[0][1]))
        vectors.append(to_upsert)
    corpus_embeddings = vectors
        #print(len())
        # upsert/insert these records to pinecone
        #index.upsert(vectors=to_upsert)

        # check that we have all vectors in index
        #index.describe_index_stats()
    #_-------------



    #print(tables[2])

# ..   corpus_embeddings,corpus_sentences = load_dataset_wiki(max_corpus_size)  ## 20.000  perguntas
# ..   print(f'dimensao do vetor: {len(corpus_embeddings[0])}')
# ..   print(f'dimensao da passage 0: {len(corpus_sentences[0])}')



    while True:
        inp_question = input("Entre com a question: ")
        num_candidates = int(input("Entre com numero para Top-K candidatos: "))

        ## 01 - similaridade da question com os embedding
        hits = cosine_similarity(inp_question,corpus_embeddings,num_candidates)

        ## 02 - re ranking com um cross-encoder
        hits = re_ranking(inp_question,corpus_sentences,hits,num_candidates)

        ## 03 - tokeniza a [pergunta:documento]
        document = corpus_sentences[hits[0]["corpus_id"]][1]   ## atencao com o [1]
        input_ids, segment_ids, tokens = get_token_from_pair(inp_question,document)   

        ## 04 - obtem a resposta
        answer = get_answer(input_ids,segment_ids)  # inputs_ids = tokens representing our input text = [pergunta:resposta]
                                                    # segment IDs = 1/0 to differentiate question from answer_text

        print(f'question    :  {inp_question}')
        print(f'documento id: {hits[0]["corpus_id"]}')
        print(f'documento title: {corpus_sentences[hits[0]["corpus_id"]][0]}')
        print(f'documento: {corpus_sentences[hits[0]["corpus_id"]][1]}')

        print(f'Answer:    {answer}')
        print('')