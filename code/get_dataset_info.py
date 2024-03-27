import os, tqdm
import pandas as pd
import pickle
import numpy as np
import json

def get_dataset_info():
        diretorio = '/data/ott-qa/new_csv'
        # Lista de arquivos no diretório
        arquivos = os.listdir(diretorio)

        # Filtrar apenas os arquivos .csv
        tabelas = [arquivo for arquivo in arquivos if arquivo.endswith('.csv')]

        print("Lista de arquivos .csv:")
        df_tabelas = pd.DataFrame()
        
        for tabela in tabelas:
            csv = os.path.join(diretorio, tabela)
            num_linhas = 0
            num_colunas = 0
            with open(csv, 'r') as Fin:
                linhas = Fin.readlines()
                        
                cabecalho = linhas[0]
                num_colunas = cabecalho.count(',') + 1
                linhas =   [linha.strip() for linha in linhas if linha.strip()]
                num_linhas = len(linhas)-1

                new_data = {'base_dir'  : diretorio,
                            'table'     : csv,
                            'num_linhas' : num_linhas,
                            'num_colunas': num_colunas,
                            'cabecalho': cabecalho
                            }
                df_tabelas = df_tabelas.append(new_data, ignore_index=True)
        
        df_tabelas.to_csv('/data/ott-qa/summary.csv', sep=',')
        print(df_tabelas['num_linhas'].mean())
        print(df_tabelas['num_linhas'].std())
        print(df_tabelas['num_linhas'].max())
        print("")
        print(df_tabelas['num_colunas'].mean())
        print(df_tabelas['num_colunas'].std())
        print(df_tabelas['num_colunas'].max())


#ver em file_name = f'/data/ott-qa/embeddings/mpnet_table_header_embeddings_{device}_{max_seq}_{max_pos}.pkl'
#numero de tokens por tabela e também por tabela com o header


def get_embeddings():
    # Some local file to cache computed embeddings
    embedding_file = f'/data/ott-qa/embeddings/mpnet_table_header_embeddings_cpu_384_514.pkl'
    device="cpu"

    # Check if embedding cache path exists
    if not os.path.exists(embedding_file):
        print(f'Embeddings nao encontrado em cache {file}')
        print('abortando')
        exit()
    print("Aguarde carregando....")

    with open(embedding_file, "rb") as fIn:
            cache_data = pickle.load(fIn)
            embeddings_dict = {
            'tables_embedding'            : cache_data["embeddings"],
            'tables_idx'                  : cache_data["tables_idx"],
            'tables_uid'                  : cache_data["tables_uid"],
            'tables_body'                 : cache_data["tables_body"],
            'tables_url'                  : cache_data["tables_url"],
            'tables_title'                : cache_data["tables_title"],
            'tables_header'               : cache_data["tables_header"],
            'tables_section_title'        : cache_data["tables_section_title"],
            'tables_section_text'         : cache_data["tables_section_text"],
            'tables_intro'                : cache_data["tables_intro"],
            'tables_tokens_len'           : cache_data["tables_tokens_len"],
            'tables_and_append_tokens_len': cache_data["tables_and_append_tokens_len"]}

    print("")
    print("Corpus loaded with {} tables / embeddings".format(len(cache_data["embeddings"])))
    print(f'dimensao do vetor : {len(cache_data["embeddings"][0])}')
    
    temp = embeddings_dict['tables_tokens_len']
    print(np.mean(temp))
    print(np.std(temp))
    print(np.max(temp))

    temp = embeddings_dict['tables_and_append_tokens_len']

    print(np.mean(temp))
    print(np.std(temp))
    print(np.max(temp))
    print(temp.index(8517))


def get_rr_output():
    file = 'mpnet_RR_table_header_cpu_384_514/oficial/0-200_mpnet_RR_table_header_cpu_384_514.json'
    
    device = 'cpu'
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
                    'rr_top_index'           : sample['rr_top_index'],
                    'rr_top_uid'             : sample['rr_top_uid'],
                    'rr_top_score'           : sample['rr_top_score'],
                    'rr_top1'                : sample['rr_top1'],
                    'rr_top10'               : sample['rr_top10'],
                    'rr_top50'               : sample['rr_top50'],
                    'rr_top100'              : sample['rr_top100'],
                    'rr_time'                : sample['rr_time'],
                    'llm_ret_column_names'   : sample['llm_ret_column_names'],   # nome colunas depois de aplicar llm na saída do retriever
                    'llm_ret_column_scores'  : sample['llm_ret_column_scores'],   # scores das colunas depois de aplicar llm na saída do retriever
                    'llm_ret_column_counts'  : sample['llm_ret_column_counts'],   # novo numero de colunas depois de aplicas llm ""       ""
                    'top_column_names'       : sample['top_column_names'],   # colunas originais
                    'top_column_counts'      : sample['top_column_counts'],   # num colunas originais

                    }
        predictions.append(new_data)
    return(predictions)





retriever_dict_list = get_rr_output()
count_errors = 0
count_acertos = 0
summary_list = []
for retriever_output in retriever_dict_list:
    colunas_list = retriever_output['llm_ret_column_names']
    scores_list = retriever_output['llm_ret_column_scores']
    for idx in range(len(colunas_list)):
        colunas = colunas_list[idx]
        scores  = scores_list[idx]
        for subidx in range(len(colunas)):
            if ((colunas[subidx] in ["page_content", "metadata"]) or (scores == -99)):
                count_errors +=1
                continue
            else:
                count_acertos +=1
                summary_list.append(scores[subidx])

frequencia = {}

# Contagem da frequência de cada valor na lista
for valor in summary_list:
    if valor in frequencia:
        frequencia[valor] += 1
    else:
        frequencia[valor] = 1

# Exibindo os valores e suas frequências
for valor, freq in frequencia.items():
    print(f"O valor {valor} aparece {freq} vezes na lista.")


print(count_errors)
