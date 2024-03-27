import pandas as pd
from tqdm import tqdm
import os
import json
#from pathlib import Path   
        
        



def get_rr_llm_output(file, device):
    rr_file = f'/data/ott-qa/output/{file}'
    # Check if file exists
    if not os.path.exists(rr_file):
        print(f'retriever file {retriever_file} nao encontrado')
        print('abortando')
        exit()
    print(f'rr file {rr_file} encontrado')
    print("Aguarde carregando....")

    with open(rr_file, 'r') as Fin:
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
                    'rr_top_index'           : sample['rr_top_index'],
                    'rr_top_uid'             : sample['rr_top_uid'],
                    'rr_top_score'           : sample['rr_top_score'],
                    'rr_top1'                : sample['rr_top1'],
                    'rr_top10'               : sample['rr_top10'],
                    'rr_top50'               : sample['rr_top50'],
                    'rr_top100'              : sample['rr_top100'],
                    'rr_time'                : sample['rr_time'],
                    'rr_column_name'         : sample['column_names'],
                    'rr_column_scores'       : sample['column_scores']
                    }
        predictions.append(new_data)

    return(predictions)

        



def main():

    device = "cpu"
    retriever_output_file = "0-500-mpnet_RR_RetriverColRel25_table_intro_cpu_384_514.json"
    rr_dict_list = get_rr_llm_output(retriever_output_file,device)
    print(len(rr_dict_list))
    THRESHOULD_RELEVANT = 0.1

    df_tabelas_list = pd.DataFrame()
    for rr_dict in tqdm(rr_dict_list):
            tables = rr_dict['top_uid']   # FAZENDO SOBRE A SAIDA DO RETRIEVER
            idx = 0
            for table in tables:
                file_name = f'/data/ott-qa/new_csv/{table}.csv'
                num_linhas = 0
                num_colunas = 0
                with open(file_name, 'r') as table_csv:
                    linhas = table_csv.readlines()
                
                cabecalho = linhas[0]
                num_colunas = cabecalho.count(',') + 1
                linhas =   [linha.strip() for linha in linhas if linha.strip()]
                num_linhas = len(linhas)-1

                print(num_linhas)
                print(num_colunas)
                print("")

                column_names  = rr_dict['rr_column_name'][idx]
                column_scores = rr_dict['rr_column_scores'][idx]

                print(column_names)
                print(column_scores)
                idx +=1

                #new_data = {'base_dir'  : input_dir,
                #            'table'      : f,
                #            'full_path'   :base_dir,
                #            'path_for_f' : path_for_f,
                #            'list'       : 1,
                #            'num_linhas' : num_linhas,
                #            'num_colunas': num_colunas,
                #            'cabecalho': cabecalho
                #            }
                #df_tabelas_list = df_tabelas_list.append(new_data, ignore_index=True)

main()
