### versao com sentença e tabela nos embeddings

from sentence_transformers import SentenceTransformer, util
import tqdm, json, os
import pandas as pd
import pickle
import torch


def calcula_metrica():
    results_retriever_files =  ['mpnet_table_cpu_512_514.json',                   # só a tabela
                                'mpnet_table_header_cpu_512_514.json',           # tabela mais a introdução do documento]
                                'mpnet_table_intro_cpu_512_514.json',          # tabela mais a o header da tabela
                                'mpnet_table_section_title_cpu_512_514.json',    # tabela mais uma passage da seção da tebale
                                'mpnet_table_section_text_cpu_512_514.json',   # tabela mais o titulo da tabela
                                'mpnet_table_cpu_384_514.json',                   # só a tabela
                                'mpnet_table_header_cpu_384_514.json',           # tabela mais a introdução do documento]
                                'mpnet_table_intro_cpu_384_514.json',          # tabela mais a o header da tabela
                                'mpnet_table_section_title_cpu_384_514.json',    # tabela mais uma passage da seção da tebale
                                'mpnet_table_section_text_cpu_384_514.json']   # tabela mais o titulo da tabela
    


    df_all = pd.DataFrame()
    data_processed = []
    for result_retriever_file in results_retriever_files:  # para cada output de cada embeddings feito
        out_put_file = "/data/ott-qa/output/" + result_retriever_file
        with open(out_put_file, "r") as f:
            data = json.load(f)

        for qa in data:
            #df_temp = pd.DataFrame()
            qa_temp = qa.copy()
            del qa_temp['predict_index']  # removendo as listas
            del qa_temp['predict_uid']
            del qa_temp['predict_score']
            qa_temp['option'] = out_put_file.split("/data/ott-qa/output/")[1]
            #df_temp = pd.DataFrame(qa_temp)
            data_processed.append(qa_temp)
        
    df_all = pd.DataFrame(data_processed)
    #df_all.append(data_processed)

    return(df_all)
    print("fim")    

def main():

    df = calcula_metrica()
    df.head()
    gb = df.groupby(['option'])
    data_list = [gb.get_group(x) for x in gb.groups]


    df_result = pd.DataFrame()
    df_all_result = pd.DataFrame()
    for data in data_list:
        data_temp = data.copy()
        data_temp = data_temp[['option','top1','top10','top50','top100']]
        df_result = data_temp.sum().to_frame().T
        df_all_result = pd.concat([df_all_result, df_result], ignore_index=True)

    
    embeddings_file =  ['mpnet table embeddings 512/514',                   # só a tabela
                        'mpnet table+intro embeddings 512/514',             # tabela mais a introdução do documento]
                        'mpnet table+header embeddings 512/514',            # tabela mais a o header da tabela
                        'mpnet table+section_text embeddings 512/514',      # tabela mais uma passage da seção da tebale
                        'mpnet table+section_title embeddings 512/514',     # tabela mais o titulo da tabela
                        'mpnet table embeddings 384/514',                   # só a tabela
                        'mpnet table+intro embeddings 384/514',             # tabela mais a introdução do documento]
                        'mpnet table+header embeddings 384/514',            # tabela mais a o header da tabela
                        'mpnet table+section_text embeddings 384/514',      # tabela mais uma passage da seção da tebale
                        'mpnet table+section_title embeddings 384/514']     # tabela mais o titulo da tabela

    #df_all_result['option'] = embeddings_file VERIFICAR

    df_all_result.top1 = df_all_result.top1/2214*100
    df_all_result.top10 = df_all_result.top10/2214*100
    df_all_result.top50 = df_all_result.top50/2214*100
    df_all_result.top100 = df_all_result.top100/2214*100
    
    
    print(df_all_result.head(10))

    media_top1 = df.loc[df['top1'] == True, 'table_tokens_len'].mean()
    media_top10 = df.loc[df['top10'] == True, 'table_tokens_len'].mean()
    media_top50 = df.loc[df['top50'] == True, 'table_tokens_len'].mean()
    media_top100 = df.loc[df['top100'] == True, 'table_tokens_len'].mean()

    print(media_top1)
    print(media_top10)
    print(media_top50)
    print(media_top100)

    media_top1 = df.loc[df['top1'] == True, 'tables_and_append_tokens_len'].mean()
    media_top10 = df.loc[df['top10'] == True, 'tables_and_append_tokens_len'].mean()
    media_top50 = df.loc[df['top50'] == True, 'tables_and_append_tokens_len'].mean()
    media_top100 = df.loc[df['top100'] == True, 'tables_and_append_tokens_len'].mean()

    print(media_top1)
    print(media_top10)
    print(media_top50)
    print(media_top100)


    print("fim")





os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


debug_mode = False

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
