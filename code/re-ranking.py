import torch
import os, json
import pickle
import time
# retriever
from sentence_transformers import SentenceTransformer, util


#retriever e re-ranker
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
import os
import csv
import pickle
import time

### inserido
import pandas as pd



def get_retriever_output(file, device):
    retriever_file = f'/data/ott-qa/output/{file}'
    # Check if file exists
    if not os.path.exists(retriever_file):
        print(f'retriever file {retriever_file} nao encontrado')
        print('abortando')
        exit()
    print(f'retriever file {retriever_file} encontrado')
    print("Aguarde carregando....")

    with open(retriever_file, 'r') as Fin:
        json_data = json.load(Fin)

    predictions = []
    for sample in json_data:
        new_data = {}
        new_data = {'question_id'            : sample['question_id'],
                    'question'               : sample['question'],
                    'table_idx'              : sample['table_idx'],
                    'table_tokens_len'       : sample['table_tokens_len'],
                    'table_tokens_append_len': sample['table_tokens_append_len'],  # aqui
                    'table_id'               : sample['table_id'],
                    'answer-text'            : sample['answer-text'],
                    'top_tokens_len'         : sample['top_tokens_len'],
                    'top_tokens_append_len'  : sample['top_tokens_append_len'],
                    'top_index'              : sample['top_index'],
                    'top_uid'                : sample['top_uid'],
                    'top_score'              : sample['top_score'],
                    'top1'                   : sample['top1'],
                    'top10'                  : sample['top10'],
                    'top50'                  : sample['top50'],
                    'top100'                 : sample['top100'],
                    'time'                   : sample['time'],
                    'rr_top_index': [],
                    'rr_top_uid'  : [],
                    'rr_top_score': [],
                    'rr_top1'         : False,
                    'rr_top10'        : False,
                    'rr_top50'        : False,
                    'rr_top100'       : False,
                    'rr_time'         : False
                    }
        predictions.append(new_data)
    return(predictions)


def re_ranking(question,tables):
    # Now, do the re-ranking with the cross-encoder
    start_time = time.time()
    sentence_pairs = [[question, table] for table in tables]  # montou os i pares pergunta:hit[i]
    cross_encoder_scores = rerank_cross_encoder_model.predict(sentence_pairs) #, show_progress_bar=True)

    #for idx in range(len(hits)):
    #    hits[idx]["cross-encoder_score"] = cross_encoder_scores[idx]

    # Sort list by CrossEncoder scores
    #hits = sorted(hits, key=lambda x: x["cross-encoder_score"], reverse=True)
    #print("\nRe-ranking with CrossEncoder took {:.3f} seconds".format(time.time() - start_time))
    

 #   for hit in hits[0:5]:
 #       print("\t{:.3f}\t{}".format(hit["cross-encoder_score"], corpus_tables[hit["corpus_id"]]))
 #       print(f'Corpus id: {hit["corpus_id"]}')
    
    return(cross_encoder_scores)

def get_top100_tables_and_headers(retriever_full_list, list_top100_id):
    tables_top100_list = []
    headers_top100_list = []
    for table_id in list_top100_id:
        #table_id = retriever_full_list[idx]['table_id']
        file_name = f'/data/ott-qa/new_csv/{table_id}.csv'
        df = pd.read_csv(file_name, sep=',')
        tables_top100_list.append(df)
        headers = df.columns.tolist()
        headers = ', '.join(headers)
        headers_top100_list.append(headers)
    return tables_top100_list, headers_top100_list



def _preprocess_tables(tables: list):
    processed = []
    # loop through all tables
    count = 0
    for table in tables:
        # convert the table to csv and
        processed_table = "\n".join([table.to_csv(index=False)])
        # add the processed table to processed list
        processed.append(processed_table)
        #if count == 10:
        #    break
        #count +=1
    return processed


def main():

    retriever_output_files =  [#'mpnet_table_cpu_512_514.json']                   # só a tabela
                               #'mpnet_table_header_cpu_512_514.json',            # tabela mais a introdução do documento]
                               #'mpnet_table_intro_cpu_512_514.json', #,             # tabela mais a o header da tabela
                               #'mpnet_table_section_title_cpu_512_514.json',     # tabela mais uma passage da seção da tebale
                               #'mpnet_table_section_text_cpu_512_514.json'] #,      # tabela mais o titulo da tabela
                               #'mpnet_table_cpu_384_514.json',    ## esses mais antigos nao tem o tables_and_append_tokens_len
                               #'mpnet_table_header_cpu_384_514.json'] #,
                               'mpnet_table_intro_cpu_384_514.json',
                               'mpnet_table_section_title_cpu_384_514.json',
                               'mpnet_table_section_text_cpu_384_514.json']




    num_candidates = 100
    # fazer aqui a leitura de todas as perguntas e a resposta

    for retriever_output_file in retriever_output_files:  # para cada retriever feito

        retriever_dict_list = get_retriever_output(retriever_output_file,device)

        count = 0
        reranking_list = []
        print(f'Processando {retriever_output_file}')
        for retriever_output in retriever_dict_list:

            #start_time = time.time()
            inp_question = retriever_output['question']
            top100_tables, top100_headers = get_top100_tables_and_headers(retriever_dict_list, retriever_output['top_uid'])
            top100_tables_processed = _preprocess_tables(top100_tables)  # nao preciso fazer append do header, já está na tabela

            scores = re_ranking(inp_question,top100_tables_processed)  # 1 question para lista das top100 tabelas
            scores = scores.tolist()

            hits = pd.DataFrame()
            hits['id'] = retriever_output['top_index'] #predict_index']
            hits['uid'] = retriever_output['top_uid']
            hits['cross-encoder_score'] = scores
            
            hits.sort_values(by='cross-encoder_score',ascending=False,ignore_index=True,inplace=True)
            

            reranking = {}
            reranking = retriever_output.copy()
            reranking['rr_top_index']= hits['id'].tolist()
            reranking['rr_top_uid']= hits['uid'].tolist()
            reranking['rr_top_score']= hits['cross-encoder_score'].tolist()

            reranking['rr_top1'] = reranking['rr_top10'] = reranking['rr_top50'] = reranking['rr_top100'] = False
            if reranking['table_idx'] == hits['id'][0]:
                reranking['rr_top1'] = True
            elif reranking['table_idx'] in hits['id'][1:11]:
                reranking['rr_top10'] = True
            elif reranking['table_idx'] in hits['id'][11:51]:
                reranking['rr_top50'] = True
            elif reranking['table_idx'] in hits['id'][51:]:
                reranking['rr_top100'] = True
            reranking_list.append(reranking)
            print(count)
            count +=1
            #break
  
        out_put_file = "/data/ott-qa/output/" + retriever_output_file
        out_put_file = out_put_file.replace("mpnet","mpnet_RR")
        print(f'criado {out_put_file}')
        #print(reranking_list[0])

        with open(out_put_file, "w") as arquivo_json:
            json.dump(reranking_list, arquivo_json)




        #    print('---------------------------------------------------------------------------'
######################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


max_corpus_size = 0 # 500 #25  # 0 significa sem restricao
#dataset_path = "data/wikipedia/simplewiki-2020-11-01.jsonl.gz"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
tables            = []
tables_embedding  = []
tables_sentence   = []
tables_title      = []
tables_file_name  = []
tables_tokens_len = []

inp_question = ''
num_candidates = 3

debug_mode = False #True
download_models = False
ConversationBuffer = True

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


    #cross_encoder_model = CrossEncoder("cross-encoder/stsb-roberta-base")
    ## reranker
    rerank_cross_encoder_model = CrossEncoder("cross-encoder/stsb-roberta-base")
    #The model will predict scores for the pairs ('Sentence 1', 'Sentence 2')



    if download_models == True:
        #retriever
        embedding_path = "/QA/Bert/data/ott-qa/embeddings/"
        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        retriever_biencoder_model = SentenceTransformer(retriever_model_name, device=device)  # para fazer o download
    
    else:
        #retriever
        embedding_path = "/data/ott-qa/embeddings/"
        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        path_local = "/modelos/deepset_all-mpnet-base-v2-table" #deepset_all-mpnet-base-v2-tablel.pth"
        retriever_biencoder_model = SentenceTransformer(path_local, device=device)

    main()







