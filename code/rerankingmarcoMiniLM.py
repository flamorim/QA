# max seq len: Specifies the maximum sequence length of one input text for the model. Mandatory.

### versao com sentença e tabela nos embeddings

from sentence_transformers import SentenceTransformer, util, CrossEncoder
import tqdm, json, os
import pandas as pd
import pickle
import torch
import re
import time, random


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



def get_top100_tables(list_top100_uid):
    tables_top100_list = []
    for table_uid in list_top100_uid:
        file_name = f'/data/ott-qa/new_csv/{table_uid}.csv'
        try:
            df = pd.read_csv(file_name, sep=',')
        except:
            df=pd.DataFrame()   ## problema no arquivo que foi retirado - gay, etc...
            print(f'error em {file_name}')
        tables_top100_list.append(df)
    return tables_top100_list

def _preprocess_tables(tables: list):
    processed = []
    # loop through all tables
    for table in tables:
        # convert the table to csv and
        processed_table = "\n".join([table.to_csv(index=False)])
        # add the processed table to processed list
        processed.append(processed_table)
    return processed

def re_ranking(question,tables):
    # Now, do the re-ranking with the cross-encoder

    diretorio = "/modelos/reranker/cross-encoder-ms-marco-MiniLM-L-6-v2"
    cross_encoder = CrossEncoder(diretorio)


    start_time = time.time()
    sentence_pairs = [[question, table] for table in tables]  # montou os i pares pergunta:hit[i]
    #cross_encoder_scores = rerank_cross_encoder_model.predict(sentence_pairs) #, show_progress_bar=True)
    # novo cross encoder
    cross_encoder_scores = cross_encoder.predict(sentence_pairs, show_progress_bar=True)

    return(cross_encoder_scores)


def build_reranking(retriever_file, device, run_improve_question):

            with open('/QA/Bert/code/path_to_files.json', 'r') as file:
                    path_to_files = json.load(file)

            dataset = pd.read_csv(retriever_file,sep=',')
            df = pd.DataFrame()

            for index, data in dataset.iterrows():
                if run_improve_question == True:
                    inp_question   = data['question_opt']
                else:
                    inp_question   = data['question_txt']

                
                top100_tables  = eval(data['top100_table_uid'])
                top100_tables_processed = get_top100_tables(top100_tables)
                top100_tables_processed = _preprocess_tables(top100_tables_processed)  # nao preciso fazer append do header, já está na tabela

                scores = re_ranking(inp_question,top100_tables_processed)  # 1 question para lista das top100 tabelas
                #scores = scores.tolist()

                rr_idx  = eval(data['top100_table_idx']) #.tolist()
                rr_uid = eval(data['top100_table_uid']) #.tolist()
                rr_cross_encoder_score = scores.tolist()
                rr = pd.DataFrame({'idx': rr_idx,  'uid': rr_uid, 'cross_encoder_score': rr_cross_encoder_score})
                rr.sort_values(by='cross_encoder_score',ascending=False,ignore_index=True,inplace=True)
            
                rr_top1 = rr_top10 = rr_top50 = rr_top100 = False
                if data['table_uid_gt'] == rr['uid'][0]:
                    rr_top1 = True
                elif data['table_uid_gt'] in rr['uid'][1:11].tolist():
                    rr_top10 = True
                elif data['table_uid_gt'] in rr['uid'][11:51].tolist():
                    rr_top50 = True
                elif data['table_uid_gt'] in rr['uid'][51:].tolist():
                    rr_top100 = True

                new_data = {'rr_top100_table_idx'   : rr['idx'].tolist(),
                            'rr_top100_table_uid'   : rr['uid'].tolist(),
                            'rr_top100_table_score' : rr['cross_encoder_score'].tolist(),
                            'rr_top1_flag'          : rr_top1,
                            'rr_top10_flag'         : rr_top10,
                            'rr_top50_flag'         : rr_top50,
                            'rr_top100_flag'        : rr_top100}
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                #df_reranking = pd.concat([df_reranking, new_data], axis=0)
                print(index)
                #    break

                if index % 500 == 0:
                    rr_file = path_to_files['reranker_marcomini_destination'] + retriever_file.split('/')[-1]
                    rr_file = rr_file.replace('csv',f'{index}.csv')
                    df_reranking = pd.concat([dataset.reset_index(drop=True), df.reset_index(drop=True)], axis=1)  ## ok
                    df_reranking.to_csv(rr_file, sep=',', index=False)       
                    print(f'criado {rr_file}')


                    #df_reranking = pd.concat([dataset.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
                    #return(df_reranking)

            df_reranking = pd.concat([dataset.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
            return(df_reranking)

if __name__ == "__main__":
    run_reranker = False
    run_improve_question = True

    if run_reranker == True:
            retriever_output = [#'mpnet_table_section_title_embeddings_cpu_512_512.csv',
                                #'mpnet_table_section_title_embeddings_cpu_384_512.csv',
                                'mpnet_table_intro_embeddings_cpu_512_512.csv',
                                'mpnet_table_embeddings_cpu_512_512.csv']
            for retriever_file in retriever_output:
                retriever_file =  f'/data/ott-qa/retriever/improved003/{retriever_file}'
                device = 'CPU'
                df_reranking = build_reranking(retriever_file, "cpu", run_improve_question)
                reranking_file = retriever_file.replace('/retriever/','/reranking/improved003/')
                df_reranking.to_csv(reranking_file,sep=',',index=False)  