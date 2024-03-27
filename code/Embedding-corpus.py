import gzip
import json
import os
#import warnings
#warnings.filterwarnings("ignore")

#retriever 
from transformers import AutoModel, AutoTokenizer
#from datasets import load_dataset
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

def _preprocess_tables(tables: list):
    processed = []
    # loop through all tables
    for table in tables:
        # convert the table to csv and
        processed_table = "\n".join([table.to_csv(index=False)])
        # add the processed table to processed list
        processed.append(processed_table)
    return processed


def get_embeddings(model_name, dataset_path, max_corpus_size):
    # Some local file to cache computed embeddings
    embedding_file = "{}-size-{}.pkl".format(model_name.replace("/", "_"), max_corpus_size)
    embedding_cache_path = dataset_path + embedding_file
    # Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):
        print(f'Embeddings nao encontrado em cache {embedding_cache_path}')
        if not os.path.exists(dataset_path):
            print(f'Dataset {dataset_path} nao encontrado local...')
            print('abortando')
            exit()
        #read_dataset!!!
        with open(dataset_path + "dataset.pkl", "rb") as fIn:
            cache_data = pickle.load(fIn)
            tables = cache_data["tables"]
            #processed_tables = tables
            csv_names = cache_data["csv_names"]
            table_sentences = cache_data["table_sentences"]
            table_captions = cache_data["table_captions"]


        processed_tables = _preprocess_tables(tables)
        corpus_embeddings = retriever_biencoder_model.encode(processed_tables, convert_to_tensor=True)
        print(f'Salvando localmente os embeddings e tabelas em {embedding_cache_path} ')
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({"tables": tables, "embeddings": corpus_embeddings}, fOut)
    else:
        print(f'Embeddings cache {embedding_cache_path} encontrado localmente...')
        print("Aguarde carregando....")

        with open(embedding_cache_path, "rb") as fIn:    ## erro gpu
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

def main():


        #retriever
        embedding_path = "/QA/Bert/data/tabfact/"
        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        path_local = "/modelos/deepset_all-mpnet-base-v2-table" #deepset_all-mpnet-base-v2-tablel.pth"
        retriever_biencoder_model = SentenceTransformer(path_local, device=device)
    
        get_embeddings(retriever_model_name, embedding_path, max_corpus_size)






if __name__ == '__main__':
    if debug_mode == True:
        import debugpy
        debugpy.listen(7011)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print('hello 2')
    i = 1  # local para breakepoint do debuger
    print('hello 3')

    main()



