import torch
import os
import pickle
import time, random
import json
import utils
import pandas as pd

import utils

from sentence_transformers import SentenceTransformer, util


#retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
path_local = "/modelos/deepset_all-mpnet-base-v2-table" #deepset_all-mpnet-base-v2-tablel.pth"
device = "cpu"
retriever_biencoder_model = SentenceTransformer(path_local, device=device)


def build_retriever(dataset_file,embedding_file, device ,run_improve_question, llm):

    with open('/QA/Bert/code/path_to_files.json', 'r') as file:
                    path_to_files = json.load(file)

    # dicionario com listas de todas as informacoes das tabelas
    list_embedding_dict = utils.get_embeddings(embedding_file,device)

    list_tables_uid            = list_embedding_dict['tables_uid']          # 8891 tabelas
    #list_tables_tokens_len     = list_embedding_dict['tables_tokens_len']  # 8891 tokens len
    list_tables_idx            = list_embedding_dict['tables_idx']          # 8891 tokens len
    #list_tables_uid            = list_embedding_dict['tables_uid']         # 8891 tokens len
    list_tables_url            = list_embedding_dict['tables_url']          # 8891 tokens len
    list_tables_header         = list_embedding_dict['tables_header']       # 8891 tokens len
    list_embeddings_tokens_len = list_embedding_dict['embedding_tokens_len']
    list_tables_intro          = list_embedding_dict['tables_intro']

    #list_tables_and_append_tokens_len = list_embedding_dict['tables_and_append_tokens_len']  # 8891

    dict_tables = {
        'uid':        list_tables_uid,
        'tokens_len': list_embeddings_tokens_len,
        'idx':        list_tables_idx,
        'url':        list_tables_url,
        'table_intro': list_tables_intro
    }
    df_tables = pd.DataFrame(dict_tables)
    df_tables['idx'] = df_tables['idx'].astype(int)
    df_tables['tokens_len'] = df_tables['tokens_len'].astype(int)
    # abrindo o dataset com as perguntas e respostas
    predictions = []
    count = 0
    with open(f'/data/{dataset_file}/released_data/dev.json', 'r') as f:    # perguntas e respostas
        data = json.load(f)
    dataset = utils.convert_dataset(data)

    #ATENCAO REMOVENDO AS QUESTIONS FILTRADAS
    #to_remove = '/QA/Bert/data/tat-qa/to_filter/remover002.unnmamed0.2024.08.10.csv'
    #df_to_remove = pd.read_csv(to_remove, sep=',')
    #list_to_remove = df_to_remove['question_id'].to_list()
    #print('antes ', dataset.shape)
    #dataset = dataset[~dataset['question_id'].isin(list_to_remove)]
    #dataset.reset_index(drop=False, inplace=False)
    #print('depois', dataset.shape)
    ########################################################################
    topk_len = 100  # tamanho da top-K

    questions = dataset.question_txt.values.tolist()

    #questions = questions[0:10]

    # 1 - calcula similaridade - so tabelas
    if run_improve_question == True:
            df_teste = pd.DataFrame()
            df_teste = pd.read_csv('/data/tat-qa/question_rewriting/improved_questions.csv',sep=',')


            #ATENCAO REMOVENDO AS QUESTIONS FILTRADAS
#            to_remove = '/QA/Bert/data/tat-qa/to_filter/remover002.unnmamed0.2024.08.10.csv'
#            df_to_remove = pd.read_csv(to_remove, sep=',')
#            list_to_remove = df_to_remove['question_id'].to_list()
#            print('antes ', df_teste.shape)
#            df_teste = df_teste[~df_teste['question_id'].isin(list_to_remove)]
#            df_teste.reset_index(drop=False, inplace=False)
#            print('depois', df_teste.shape)
            ########################################################################





            questions_opt = df_teste['questions_opt'].to_list()
            questions_opt_status = df_teste['questions_opt_status'].to_list()
            #questions_opt,questions_opt_status = llmquestions.build_improve_questions(questions, device='CPU', llm=llm)
            hits = utils.cosine_similarity(retriever_biencoder_model, questions_opt, list_embedding_dict['embeddings'], topk_len)
    else:
            hits = utils.cosine_similarity(retriever_biencoder_model, questions, list_embedding_dict['embeddings'], topk_len)

    df_hits = utils.convert_to_df(hits)  # retorna df com listas top-100 de idx e score 
    dataset['top100_table_idx']   = df_hits['idx']
    dataset['top100_table_score'] = df_hits['score']

    # Inicializar uma lista para armazenar os tokens correspondentes
    table_tokens_len_list = []
    table_uid_list        = []
    table_idx_gt = []
    table_url_gt = []
    table_header_gt = []
    table_token_len_gt = []
    top1_flag_list = []
    top10_flag_list = []
    top50_flag_list = []
    top100_flag_list = []
    table_intro_list = []

    #df_hits.top1 = df_hits.top10 = df_hits.top50 = df_hits.top100 = False
    index = 0
    # Iterar sobre cada lista top100 do dataset de perguntas
    for list_top100_table_idx, answer_table_uid in zip(dataset['top100_table_idx'],dataset['table_uid_gt']):
            # Inicializar uma lista para armazenar os tokens correspondentes para a lista de id atual
            tokens_len_top100 = []
            table_uid_top100  = []
            table_intro_top100= []
            # Iterar sobre cada id na lista atual
            for id_item in list_top100_table_idx:
                # Encontrar o token correspondente na df_tables para o id atual
                tokens_len = df_tables.loc[df_tables['idx'] == id_item, 'tokens_len'].values
                uids       = df_tables.loc[df_tables['idx'] == id_item, 'uid'].values
                intros     = df_tables.loc[df_tables['idx'] == id_item, 'table_intro'].values
                
                tokens_len_top100.append(tokens_len[0])
                table_uid_top100.append(uids[0])
                table_intro_top100.append(intros[0])

            table_intro_list.append(table_intro_top100)
            table_tokens_len_list.append(tokens_len_top100)
            table_uid_list.append(table_uid_top100)
            try:
                idx_gt = list_tables_uid.index(f'/data/tat-qa/csv/{answer_table_uid}.csv') 
                table_idx_gt.append(idx_gt)
                table_url_gt.append(list_tables_url[idx_gt])
                table_token_len_gt.append(table_tokens_len_list[index][0])  ### verificar
                table_header_gt.append(list_tables_header[idx_gt])
            except:
                table_idx_gt.append(9999)
                table_url_gt.append("error")
                table_token_len_gt.append(9999)
                table_header_gt.append("error")
                print("sem tabela gt")


            #answer_table_id = dataset.loc[]
            #answer_table_uid = lista_id['table_uid'].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.').replace('*', '')  # ground trhuth
            #answer_table_idx = dataset

            if f'/data/tat-qa/csv/{answer_table_uid}.csv' == table_uid_top100[0]:
                top1_flag_list.append(True)
                top10_flag_list.append(False)
                top50_flag_list.append(False)
                top100_flag_list.append(False)
            elif f'/data/tat-qa/csv/{answer_table_uid}.csv' in table_uid_top100[1:11]:   # ATENCAO PARA A ESTATISTICA
                top1_flag_list.append(False)
                top10_flag_list.append(True)
                top50_flag_list.append(False)
                top100_flag_list.append(False)
            elif f'/data/tat-qa/csv/{answer_table_uid}.csv' in table_uid_top100[11:51]:
                top1_flag_list.append(False)
                top10_flag_list.append(False)
                top50_flag_list.append(True)
                top100_flag_list.append(False)
            elif f'/data/tat-qa/csv/{answer_table_uid}.csv' in table_uid_top100[51:100]:
                top1_flag_list.append(False)
                top10_flag_list.append(False)
                top50_flag_list.append(False)
                top100_flag_list.append(True)
            else:
                top1_flag_list.append(False)
                top10_flag_list.append(False)
                top50_flag_list.append(False)
                top100_flag_list.append(False)
            index += 1
            count += 1
            print(count)
            if run_improve_question == True:
                dataset['question_opt']    = questions_opt
                dataset['question_opt_status']    = questions_opt_status


            if index % 50 == 0:
                retriever_file = path_to_files['retriever_destination'] + embedding_file.split('/')[-1]
                retriever_file = retriever_file.replace('.pkl','.csv')
                retriever_file = retriever_file.replace('csv',f'{index}.csv')

                #retriever_file = embedding_file.replace('/embeddings/','/retriever/').replace('.pkl','.csv')
                #if run_improve_question == True:
                #    retriever_file = retriever_file.replace('csv',f'{index}.csv')
                #    retriever_file = retriever_file.replace('improved','improved003')
                dataset.to_csv(retriever_file,sep=',',index=False)
                print(f'salvo arquivo {retriever_file}')



    dataset['table_idx_gt']    = table_idx_gt
    dataset['table_url_gt']    = table_url_gt
    dataset['table_token_len_gt'] = table_token_len_gt
    dataset['table_header_gt'] = table_header_gt
    dataset['top100_table_token_len'] = table_tokens_len_list   # conferir - top100
    dataset['top100_table_uid']       = table_uid_list
    dataset['top1_flag']       = top1_flag_list
    dataset['top10_flag']      = top10_flag_list
    dataset['top50_flag']      = top50_flag_list
    dataset['top100_flag']     = top100_flag_list
    dataset['top100_table_intro'] = table_intro_list    

    return dataset


def remove_questions(file_source, to_remove):
    df = pd.read_csv(file_source, sep=',')
    df_to_remove = pd.read_csv(to_remove, sep=',')
    list_to_remove = df_to_remove['question_id'].to_list()

    print('antes ', df.shape)
    df_filtered = df[~df['question_id'].isin(list_to_remove)]
    print('depois ', df_filtered.shape)

    #file_source = file_source.replace("improved","improved/improved_e_filter002")

    new_file = file_source.replace(".csv","_filtered.csv")
    df_filtered.to_csv(new_file, sep=',', index=False)


 
#fin = '/data/tat-qa/retriever/mpnet_table_intro_embeddings_cpu_512_512.csv'
#to_remove = '/data/tat-qa/remover.2024.07.30.csv'
#to_remove = '/data/tat-qa/to_filter/remover002.unnmamed0.2024.08.10.csv'
#remove_questions(fin, to_remove)