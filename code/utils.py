import os
import json
import time
import pickle
import pandas as pd
# versao GPU
# from langchain.document_loaders import CSVLoader, DataFrameLoader
from sentence_transformers import util ## para o retriever


from langchain.document_loaders import CSVLoader


## funcoes do retriever
## primeira
def get_embeddings(file, device):
    # Some local file to cache computed embeddings

    start_time = time.time()
    
    # Check if embedding cache path exists
    if not os.path.exists(file):
        print(f'Embeddings nao encontrado em cache {file}')
        print('abortando')
        exit()
    print(f'Embeddings cache {file} encontrado localmente...')
    print("Aguarde carregando....")

    with open(file, "rb") as fIn:
            cache_data = pickle.load(fIn)
            embeddings_dict = {
            'embeddings'                  : cache_data["embeddings"],#[0:10],
            'tables_idx'                  : cache_data["tables_idx"],#[0:10],
            'tables_uid'                  : cache_data["tables_uid"],#[0:10],
            'tables_body'                 : cache_data["tables_body"],#[0:10],
            'tables_url'                  : cache_data["tables_url"],#[0:10],
            'tables_title'                : cache_data["tables_title"],#[0:10],
            'tables_header'               : cache_data["tables_header"],#[0:10],
            'tables_section_title'        : cache_data["tables_section_title"],#[0:10],
            'tables_section_text'         : cache_data["tables_section_text"],#[0:10],
            'tables_intro'                : cache_data["tables_intro"],#[0:10],
            'embedding_tokens_len'        : cache_data["embedding_tokens_len"]}#[0:10]}
#            'tables_and_append_tokens_len': cache_data["tables_and_append_tokens_len"]}

    print("")
    print("Corpus loaded with {} tables / embeddings".format(len(cache_data["embeddings"])))
    print(f'dimensao do vetor : {len(cache_data["embeddings"][0])}')
    print(f'Table + sentences embeddings load took {(time.time() - start_time)} seconds')

    return embeddings_dict

# segunda
def cosine_similarity(retriever_biencoder_model, inp_question, corpus_embeddings, num_candidates):

    start_time = time.time()
    question_embedding = retriever_biencoder_model.encode(inp_question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=num_candidates)
    #hits = hits[0]  # Get the hits for the first query
    #print("")
    #print('---------------------------------------------------------------------------')
    #print(f'Cosine-Similarity para top-{num_candidates} levou {(time.time() - start_time)} seconds')
    #print(f'imprimindo Top 5 dos {num_candidates} hits with cosine-similarity:')
    #for hit in hits[0:5]:
    #    print("\t{:.3f}\t{}".format(hit["score"], corpus_tables[hit["corpus_id"]]))

    return hits

# terceira
# transforma as listas em um df
def convert_dataset(data):
        
        question_id     = []
        question_txt    = []
        #table_idx_gt    = []
        table_uid_gt    = []
        answer_text_gt  = []
        #question_postag = []
        
        idx = 0
        for qa in data:
            question_id.append(qa["question_id"])
            question_txt.append(qa["question"])
            table_uid_gt.append(qa["table_id"].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.').replace('*', ''))  # ground trhuth

            #table_idx_gt.append(idx)
            answer_text_gt.append(qa["answer-text"])
            #question_postag.append(qa["question_postag"])
            idx +=1

        df = pd.DataFrame({'question_id'     : question_id,
                           'question_txt'    : question_txt,
                           'table_uid_gt'    : table_uid_gt,
                           #'table_idx_gt'    : table_idx_gt,
                           'answer_text_gt'  : answer_text_gt})
                           #'question_postag' : question_postag})
        
        df.top1_flag   = False
        df.top10_flag  = False
        df.top50_flag  = False
        df.top100_flag = False

        return(df)
        
def convert_to_df(hits):
# Transformar a lista em um DataFrame
    df = pd.DataFrame([(hit[0]["corpus_id"], hit[1]["corpus_id"], hit[2]["corpus_id"], hit[3]["corpus_id"], hit[4]["corpus_id"],
                        hit[5]["corpus_id"], hit[6]["corpus_id"], hit[7]["corpus_id"], hit[8]["corpus_id"], hit[9]["corpus_id"],
                        hit[10]["corpus_id"], hit[11]["corpus_id"], hit[12]["corpus_id"], hit[13]["corpus_id"], hit[14]["corpus_id"],
                        hit[15]["corpus_id"], hit[16]["corpus_id"], hit[17]["corpus_id"], hit[18]["corpus_id"], hit[19]["corpus_id"],
                        hit[20]["corpus_id"], hit[21]["corpus_id"], hit[22]["corpus_id"], hit[23]["corpus_id"], hit[24]["corpus_id"],
                        hit[25]["corpus_id"], hit[26]["corpus_id"], hit[27]["corpus_id"], hit[28]["corpus_id"], hit[29]["corpus_id"],
                        hit[30]["corpus_id"], hit[31]["corpus_id"], hit[32]["corpus_id"], hit[33]["corpus_id"], hit[34]["corpus_id"],
                        hit[35]["corpus_id"], hit[36]["corpus_id"], hit[37]["corpus_id"], hit[38]["corpus_id"], hit[39]["corpus_id"],
                        hit[40]["corpus_id"], hit[41]["corpus_id"], hit[42]["corpus_id"], hit[43]["corpus_id"], hit[44]["corpus_id"],
                        hit[45]["corpus_id"], hit[46]["corpus_id"], hit[47]["corpus_id"], hit[48]["corpus_id"], hit[49]["corpus_id"],
                        hit[50]["corpus_id"], hit[51]["corpus_id"], hit[52]["corpus_id"], hit[53]["corpus_id"], hit[54]["corpus_id"],
                        hit[55]["corpus_id"], hit[56]["corpus_id"], hit[57]["corpus_id"], hit[58]["corpus_id"], hit[59]["corpus_id"],
                        hit[60]["corpus_id"], hit[61]["corpus_id"], hit[62]["corpus_id"], hit[63]["corpus_id"], hit[64]["corpus_id"],
                        hit[65]["corpus_id"], hit[66]["corpus_id"], hit[67]["corpus_id"], hit[68]["corpus_id"], hit[69]["corpus_id"],
                        hit[70]["corpus_id"], hit[71]["corpus_id"], hit[72]["corpus_id"], hit[73]["corpus_id"], hit[74]["corpus_id"],
                        hit[75]["corpus_id"], hit[76]["corpus_id"], hit[77]["corpus_id"], hit[78]["corpus_id"], hit[79]["corpus_id"],
                        hit[80]["corpus_id"], hit[81]["corpus_id"], hit[82]["corpus_id"], hit[83]["corpus_id"], hit[84]["corpus_id"],
                        hit[85]["corpus_id"], hit[86]["corpus_id"], hit[87]["corpus_id"], hit[88]["corpus_id"], hit[89]["corpus_id"],
                        hit[90]["corpus_id"], hit[91]["corpus_id"], hit[92]["corpus_id"], hit[93]["corpus_id"], hit[94]["corpus_id"],
                        hit[95]["corpus_id"], hit[96]["corpus_id"], hit[97]["corpus_id"], hit[98]["corpus_id"], hit[99]["corpus_id"],

                        
                        hit[0]["score"], hit[1]["score"], hit[2]["score"], hit[3]["score"], hit[4]["score"], 
                        hit[5]["score"], hit[6]["score"], hit[7]["score"], hit[8]["score"], hit[9]["score"],
                        hit[10]["score"], hit[11]["score"], hit[12]["score"], hit[13]["score"], hit[14]["score"], 
                        hit[15]["score"], hit[16]["score"], hit[17]["score"], hit[18]["score"], hit[19]["score"], 
                        hit[20]["score"], hit[21]["score"], hit[22]["score"], hit[23]["score"], hit[24]["score"], 
                        hit[25]["score"], hit[26]["score"], hit[27]["score"], hit[28]["score"], hit[29]["score"], 
                        hit[30]["score"], hit[31]["score"], hit[32]["score"], hit[33]["score"], hit[34]["score"], 
                        hit[35]["score"], hit[36]["score"], hit[37]["score"], hit[38]["score"], hit[39]["score"], 
                        hit[40]["score"], hit[41]["score"], hit[42]["score"], hit[43]["score"], hit[44]["score"], 
                        hit[45]["score"], hit[46]["score"], hit[47]["score"], hit[48]["score"], hit[49]["score"],
                        hit[50]["score"], hit[51]["score"], hit[52]["score"], hit[53]["score"], hit[54]["score"], 
                        hit[55]["score"], hit[56]["score"], hit[57]["score"], hit[58]["score"], hit[59]["score"], 
                        hit[60]["score"], hit[61]["score"], hit[62]["score"], hit[63]["score"], hit[64]["score"], 
                        hit[65]["score"], hit[66]["score"], hit[67]["score"], hit[68]["score"], hit[69]["score"], 
                        hit[70]["score"], hit[71]["score"], hit[72]["score"], hit[73]["score"], hit[74]["score"], 
                        hit[75]["score"], hit[76]["score"], hit[77]["score"], hit[78]["score"], hit[79]["score"], 
                        hit[80]["score"], hit[81]["score"], hit[82]["score"], hit[83]["score"], hit[84]["score"], 
                        hit[85]["score"], hit[86]["score"], hit[87]["score"], hit[88]["score"], hit[89]["score"],
                        hit[90]["score"], hit[91]["score"], hit[92]["score"], hit[93]["score"], hit[94]["score"], 
                        hit[95]["score"], hit[96]["score"], hit[97]["score"], hit[98]["score"], hit[99]["score"])
                        
                        
                        
                        
                        
                        for hit in hits],
                    columns=['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10',
                             'id11', 'id12', 'id13', 'id14', 'id15', 'id16', 'id17', 'id18', 'id19', 'id20',
                             'id21', 'id22', 'id23', 'id24', 'id25', 'id26', 'id27', 'id28', 'id29', 'id30',
                             'id31', 'id32', 'id33', 'id34', 'id35', 'id36', 'id37', 'id38', 'id39', 'id40',
                             'id41', 'id42', 'id43', 'id44', 'id45', 'id46', 'id47', 'id48', 'id49', 'id50',
                             'id51', 'id52', 'id53', 'id54', 'id55', 'id56', 'id57', 'id58', 'id59', 'id60',
                             'id61', 'id62', 'id63', 'id64', 'id65', 'id66', 'id67', 'id68', 'id69', 'id70',
                             'id71', 'id72', 'id73', 'id74', 'id75', 'id76', 'id77', 'id78', 'id79', 'id80',
                             'id81', 'id82', 'id83', 'id84', 'id85', 'id86', 'id87', 'id88', 'id89', 'id90',
                             'id91', 'id92', 'id93', 'id94', 'id95', 'id96', 'id97', 'id98', 'id99', 'id100',

                            'score1', 'score2', 'score3', 'score4', 'score5', 'score6', 'score7', 'score8', 'score9', 'score10',
                            'score11', 'score12', 'score13', 'score14', 'score15', 'score16', 'score17', 'score18', 'score19', 'score20',
                            'score21', 'score22', 'score23', 'score24', 'score25', 'score26', 'score27', 'score28', 'score29', 'score30',
                            'score31', 'score32', 'score33', 'score34', 'score35', 'score36', 'score37', 'score38', 'score39', 'score40',                          
                            'score41', 'score42', 'score43', 'score44', 'score45', 'score46', 'score47', 'score48', 'score49', 'score50',                            
                            'score51', 'score52', 'score53', 'score54', 'score55', 'score56', 'score57', 'score58', 'score59', 'score60',                            
                            'score61', 'score62', 'score63', 'score64', 'score65', 'score66', 'score67', 'score68', 'score69', 'score70',                            
                            'score71', 'score72', 'score73', 'score74', 'score75', 'score76', 'score77', 'score78', 'score79', 'score80',                            
                            'score81', 'score82', 'score83', 'score84', 'score85', 'score86', 'score87', 'score88', 'score89', 'score90',                            
                            'score91', 'score92', 'score93', 'score94', 'score95', 'score96', 'score97', 'score98', 'score99', 'score100',                            
                            ])

    # Juntar os IDs e Scores em uma única coluna
    df_converted = pd.DataFrame()
    df_converted['idx'] = df[['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10',
    'id11', 'id12', 'id13', 'id14', 'id15', 'id16', 'id17', 'id18', 'id19', 'id20',
    'id21', 'id22', 'id23', 'id24', 'id25', 'id26', 'id27', 'id28', 'id29', 'id30',
    'id31', 'id32', 'id33', 'id34', 'id35', 'id36', 'id37', 'id38', 'id39', 'id40',
    'id41', 'id42', 'id43', 'id44', 'id45', 'id46', 'id47', 'id48', 'id49', 'id50',
    'id51', 'id52', 'id53', 'id54', 'id55', 'id56', 'id57', 'id58', 'id59', 'id60',
    'id61', 'id62', 'id63', 'id64', 'id65', 'id66', 'id67', 'id68', 'id69', 'id70',
    'id71', 'id72', 'id73', 'id74', 'id75', 'id76', 'id77', 'id78', 'id79', 'id80',
    'id81', 'id82', 'id83', 'id84', 'id85', 'id86', 'id87', 'id88', 'id89', 'id90',
    'id91', 'id92', 'id93', 'id94', 'id95', 'id96', 'id97', 'id98', 'id99', 'id100',]].values.tolist()

    df_converted['score'] = df[['score1', 'score2', 'score3', 'score4', 'score5', 'score6', 'score7', 'score8', 'score9', 'score10',
    'score11', 'score12', 'score13', 'score14', 'score15', 'score16', 'score17', 'score18', 'score19', 'score20',
    'score21', 'score22', 'score23', 'score24', 'score25', 'score26', 'score27', 'score28', 'score29', 'score30',
    'score31', 'score32', 'score33', 'score34', 'score35', 'score36', 'score37', 'score38', 'score39', 'score40',
    'score41', 'score42', 'score43', 'score44', 'score45', 'score46', 'score47', 'score48', 'score49', 'score50',
    'score51', 'score52', 'score53', 'score54', 'score55', 'score56', 'score57', 'score58', 'score59', 'score60',
    'score61', 'score62', 'score63', 'score64', 'score65', 'score66', 'score67', 'score68', 'score69', 'score70',
    'score71', 'score72', 'score73', 'score74', 'score75', 'score76', 'score77', 'score78', 'score79', 'score80',
    'score81', 'score82', 'score83', 'score84', 'score85', 'score86', 'score87', 'score88', 'score89', 'score90',
    'score91', 'score92', 'score93', 'score94', 'score95', 'score96', 'score97', 'score98', 'score99', 'score100']].values.tolist()

    #print(df)
    return(df_converted)

# transforma as listas em um df
def convert_dataset(data):
        question_idx    = []
        question_id     = []
        question_txt    = []
        #table_idx_gt    = []
        table_uid_gt    = []
        answer_text_gt  = []
        answer_type     = []
        answer_from     = []

        #question_postag = []
        
        idx = 0
        for qa in data:
            question_id.append(qa["question_id"])
            question_txt.append(qa["question"])
            table_uid_gt.append(qa["table_id"].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.').replace('*', ''))  # ground trhuth

            #table_idx_gt.append(idx)
            answer_text_gt.append(qa["answer-text"])
            answer_type.append(qa['answer_type'])
            answer_from.append(qa['answer_from'])
            
            question_idx.append(idx)

            #question_postag.append(qa["question_postag"])
            idx +=1

        df = pd.DataFrame({'question_idx'     : question_idx,
                           'question_id'     : question_id,
                           'question_txt'    : question_txt,
                           'table_uid_gt'    : table_uid_gt,
                           #'table_idx_gt'    : table_idx_gt,
                           'answer_text_gt'  : answer_text_gt,
                           'answer_type'     : answer_type,
                           'answer_from'     : answer_from})
                           #'question_postag' : question_postag})
        
        df.top1_flag   = False
        df.top10_flag  = False
        df.top50_flag  = False
        df.top100_flag = False

        return(df)

## fim funcoes do retriever
#     
##
#uncoes do reranker


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


# fim

def get_rr_output(file, device):
    retriever_file = f'/data/ott-qa/output/mpnet_RR_table_header_cpu_384_514/oficial/{file}'

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
                    'table_idx'              : sample['table_idx'],                 # ground thruth
                    'table_tokens_len'       : sample['table_tokens_len'],
                    'table_tokens_append_len': sample['table_tokens_append_len'],
                    'table_id'               : sample['table_id'],                  # ground thruth
                    'answer-text'            : sample['answer-text'],               # ground thruth
                    'top_tokens_len'         : sample['top_tokens_len'],
                    'top_tokens_append_len'  : sample['top_tokens_append_len'],
                    'top_index'              : sample['top_index'],                 # lista de saida do retriever
                    'top_uid'                : sample['top_uid'],                   # lista de saida do retriever
                    'top_score'              : sample['top_score'],                 # lista de saida do retriever
                    'top1'                   : sample['top1'],                      # true/false
                    'top10'                  : sample['top10'],                     # true/false
                    'top50'                  : sample['top50'],                     # true/false
                    'top100'                 : sample['top100'],                    # true/false
                    'time'                   : sample['time'],
                    'rr_top_index'           : sample['rr_top_index'],
                    'rr_top_uid'             : sample['rr_top_uid'],
                    'rr_top_score'           : sample['rr_top_score'],
                    'rr_top1'                : sample['rr_top1'],
                    'rr_top10'               : sample['rr_top10'],
                    'rr_top50'               : sample['rr_top50'],
                    'rr_top100'              : sample['rr_top100'],
                    'rr_time'                : sample['rr_time'],
                    'llm_ret_column_names'   : sample['llm_ret_column_names'],    # nome colunas depois de aplicar llm na saída do retriever
                    'llm_ret_column_scores'  : sample['llm_ret_column_scores'],   # scores das colunas depois de aplicar llm na saída do retriever
                    'llm_ret_column_counts'  : sample['llm_ret_column_counts'],   # novo numero de colunas depois de aplicas llm ""       ""
                    'top_column_names'       : sample['top_column_names'],        # colunas originais
                    'top_column_counts'      : sample['top_column_counts'],       # num colunas originais
                    'tapex_answer'           : [],
                    'tapas_answer'           : [],
                  }
        predictions.append(new_data)
    return(predictions)
    

def get_reader_output(file):
    reader_file = f'/data/ott-qa/output/mpnet_RR_table_header_cpu_384_514/reader/{file}'
    # Check if file exists
    if not os.path.exists(reader_file):
        print(f'retriever file {reader_file} nao encontrado')
        print('abortando')
        exit()
    print(f'retriever file {reader_file} encontrado')
    print("Aguarde carregando....")

    with open(reader_file, 'r') as Fin:
        json_data = json.load(Fin)

    predictions = []
    for sample in json_data:
        new_data = {}
        new_data = {'question_id'            : sample['question_id'],
                    'question'               : sample['question'],
                    'table_idx'              : sample['table_idx'],                 # ground thruth
                    'table_tokens_len'       : sample['table_tokens_len'],
                    'table_tokens_append_len': sample['table_tokens_append_len'],
                    'table_id'               : sample['table_id'],                  # ground thruth
                    'answer-text'            : sample['answer-text'],               # ground thruth
                    'top_tokens_len'         : sample['top_tokens_len'],
                    'top_tokens_append_len'  : sample['top_tokens_append_len'],
                    'top_index'              : sample['top_index'],                 # lista de saida do retriever
                    'top_uid'                : sample['top_uid'],                   # lista de saida do retriever
                    'top_score'              : sample['top_score'],                 # lista de saida do retriever
                    'top1'                   : sample['top1'],                      # true/false
                    'top10'                  : sample['top10'],                     # true/false
                    'top50'                  : sample['top50'],                     # true/false
                    'top100'                 : sample['top100'],                    # true/false
                    'time'                   : sample['time'],
                    'rr_top_index'           : sample['rr_top_index'],
                    'rr_top_uid'             : sample['rr_top_uid'],
                    'rr_top_score'           : sample['rr_top_score'],
                    'rr_top1'                : sample['rr_top1'],
                    'rr_top10'               : sample['rr_top10'],
                    'rr_top50'               : sample['rr_top50'],
                    'rr_top100'              : sample['rr_top100'],
                    'rr_time'                : sample['rr_time'],
                    'llm_ret_column_names'   : sample['llm_ret_column_names'],    # nome colunas depois de aplicar llm na saída do retriever
                    'llm_ret_column_scores'  : sample['llm_ret_column_scores'],   # scores das colunas depois de aplicar llm na saída do retriever
                    'llm_ret_column_counts'  : sample['llm_ret_column_counts'],   # novo numero de colunas depois de aplicas llm ""       ""
                    'top_column_names'       : sample['top_column_names'],        # colunas originais
                    'top_column_counts'      : sample['top_column_counts'],       # num colunas originais
                    'tapex_answer'           : sample['tapex_answer'],    ## nao usado
                    'tapas_answer'           : sample['tapas_answer'],    ## nao usado
                    'tapex_baseline_answer-text': sample['tapex_baseline_answer-text'],
                    'tapex_baseline_score'   :    sample['tapex_baseline_score'],  # true or false se acertou
                    'tapas_baseline_answer-text': sample['tapas_baseline_answer-text'],
                    'tapas_baseline_score'   :    sample['tapas_baseline_score']  # true or false se acertou

                  }
        predictions.append(new_data)
    return(predictions)


def get_rr_output(file, device):
    retriever_file = f'/data/ott-qa/output/mpnet_RR_table_header_cpu_384_514/oficial/{file}'

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
                    'table_idx'              : sample['table_idx'],                 # ground thruth
                    'table_tokens_len'       : sample['table_tokens_len'],
                    'table_tokens_append_len': sample['table_tokens_append_len'],
                    'table_id'               : sample['table_id'],                  # ground thruth
                    'answer-text'            : sample['answer-text'],               # ground thruth
                    'top_tokens_len'         : sample['top_tokens_len'],
                    'top_tokens_append_len'  : sample['top_tokens_append_len'],
                    'top_index'              : sample['top_index'],                 # lista de saida do retriever
                    'top_uid'                : sample['top_uid'],                   # lista de saida do retriever
                    'top_score'              : sample['top_score'],                 # lista de saida do retriever
                    'top1'                   : sample['top1'],                      # true/false
                    'top10'                  : sample['top10'],                     # true/false
                    'top50'                  : sample['top50'],                     # true/false
                    'top100'                 : sample['top100'],                    # true/false
                    'time'                   : sample['time'],
                    'rr_top_index'           : sample['rr_top_index'],
                    'rr_top_uid'             : sample['rr_top_uid'],
                    'rr_top_score'           : sample['rr_top_score'],
                    'rr_top1'                : sample['rr_top1'],
                    'rr_top10'               : sample['rr_top10'],
                    'rr_top50'               : sample['rr_top50'],
                    'rr_top100'              : sample['rr_top100'],
                    'rr_time'                : sample['rr_time'],
                    'llm_ret_column_names'   : sample['llm_ret_column_names'],    # nome colunas depois de aplicar llm na saída do retriever
                    'llm_ret_column_scores'  : sample['llm_ret_column_scores'],   # scores das colunas depois de aplicar llm na saída do retriever
                    'llm_ret_column_counts'  : sample['llm_ret_column_counts'],   # novo numero de colunas depois de aplicas llm ""       ""
                    'top_column_names'       : sample['top_column_names'],        # colunas originais
                    'top_column_counts'      : sample['top_column_counts'],       # num colunas originais
                    'tapex_answer'           : [],
                    'tapas_answer'           : [],
                  }
        predictions.append(new_data)
    return(predictions)

def build_template_reader(num_tables):

    docs = ''
    for count in range(num_tables):
        doc = f"doc{count+1}"
        doc = "{" + doc + "},"
        docs = docs + doc
    docs = docs[0:-1]

    #scores = ''
    #for count in range(num_tables):
    #    score = f"score{count+1}"
    #    score = "{" + score + "},"
    #    scores = scores + score
    #scores = scores[0:-1]

    #Each table has a score showing how relevant the table is about the question. \
    #To show how relevant the table is about the question, you will receive tables scores that is delimited by triple backticks \

    question = "{inp_question}"
    prompt_template = f"""You are a very smart oil and gas engineer working at Petrobras in Brazil.\
            Your task is generate a technical and comprehensive answer to questions that is delimited by triple backticks \
            
            The answer must be based on the tables you will receive that are delimited by triple backticks. \
            
            First you have to translate the question to english <<<question_english>>> and answer the question in english. \
            For each table check whether it is related to the question. \
            Only use tables that are related to the question to answer it.  \
            Ignore tables that are not related to the question. \
            If the answer exists in several tables, summarize them. \

            Always use references in the form [NUMBER OF DOCUMENT] when using information from a document. e.g. [3], for Document[3], \
            and write the document file name in <<<source>>>. e.g. taba.html.csv, for Document taba.html.csv
            The reference must only refer to the number that comes in square brackets after passage. \
            Otherwise, do not use brackets in your answer and reference ONLY the number of the passage without mentioning the word passage. \

            If the tables can't answer the question or you are unsure say: 'The answer can't be found in the text'. \

            You will try your best to write a concise answer in a didactic yet detailed way, \
            being truthful to the original tables. \

            Do not add any other information besides the <<<question>>>, <<<question_english>>>, <<<table_name>>>, <<<file_name>>> and <<<answer>>>.  \
            Only answer based on the tables provided. Don't make things up. \

            Format the output as JSON with the following keys: \
            question: <<<question>>> \
            question_english: <<<question_english>>> \
            answer:    <<<answer>>>  \
            source:    <<<source>>>  \

            These are the question and the tables: \
            question: ```{question}``` \
            tables:   ```{docs}``` \

            """

    return(prompt_template)


def build_input_reader(inp_question, tables):
    # poderia usar o CSVLoader, mas nessa ja tinha o df pronto
    input_for_chain = {}
    input_for_chain['inp_question'] = inp_question

    for table in tables:
        #loader = CSVLoader(file_path=table,csv_args={"delimiter": ","})
        loader = DataFrameLoader(table)
        table = loader.load()          
        table_list.append(table)
    input_for_chain['table'] = table_list

    return(input_for_chain)



def build_template_new(num_tables):

    docs = ''
    for count in range(num_tables):
        doc = f"doc{count+1}"
        doc = "{" + doc + "},"
        docs = docs + doc
    docs = docs[0:-1]

    scores = ''
    for count in range(num_tables):
        score = f"score{count+1}"
        score = "{" + score + "},"
        scores = scores + score
    scores = scores[0:-1]
    #Each table has a score showing how relevant the table is about the question. \
    #To show how relevant the table is about the question, you will receive tables scores that is delimited by triple backticks \
    #reader_answer: ```{reader_answer}```

    question = "{inp_question}"
    prompt_template = f"""You are a very smart oil and gas engineer working at Petrobras in Brazil.\
            Your task is generate a technical and comprehensive answer to questions that are delimited by triple backticks \
            
            The answer must be based on the tables you will receive that are delimited by triple backticks. \
            
            First you have to translate the question to english <<<question_english>>> and answer the question in english. \
            For each table check whether it is related to the question. \
            Each table has a score showing how relevant the table is about the question. \
            To show how relevant the table is about the question, you will receive tables scores that is delimited by triple backticks \

            Only use tables that are related to the question to answer it.  \
            Ignore tables that are not related to the question. \
            If the answer exists in several tables, summarize them. \

            Always use references in the form [NUMBER OF DOCUMENT] when using information from a document. e.g. [3], for Document[3], \
            and write the document file name in <<<source>>>. e.g. taba.html.csv, for Document taba.html.csv
            The reference must only refer to the number that comes in square brackets after passage. \
            Otherwise, do not use brackets in your answer and reference ONLY the number of the passage without mentioning the word passage. \

            If the tables can't answer the question or you are unsure say: 'The answer can't be found in the text'. \

            You will try your best to write a concise answer in a didactic yet detailed way, \
            being truthful to the original tables. \

            Do not add any other information besides the <<<question>>>, <<<question_english>>>, <<<table_name>>>, <<<file_name>>> and <<<answer>>>.  \
            Only answer based on the tables provided. Don't make things up. \

            Format the output as JSON with the following keys: \
            question: <<<question>>> \
            question_english: <<<question_english>>> \
            answer:    <<<answer>>>  \
            source:    <<<source>>>  \

            These are the question, the tables and the candidate answer: \
            question: ```{question}``` \
            tables:   ```{docs}```     \
            scores:   ```{scores}```   \
         

            """

    return(prompt_template)


def build_input_evaluation(reader_dict):
    input_for_chain = {}
    input_for_chain['inp_question'] = reader_dict['inp_question']
    input_for_chain['answerone']       = reader_dict['answerone']
    input_for_chain['answertwo']       = reader_dict['answertwo']
    
    tb = "/data/ott-qa/new_csv/" + reader_dict['knowledge'] + ".csv"

    loader = CSVLoader(file_path=tb, csv_args={"delimiter": ","})
    try:
        doc = loader.load()
    except:
        doc = ""
          
    input_for_chain['knowledge'] = doc

    return(input_for_chain)



def build_input_rag(reader_dict, topk):
    input_for_chain = {}
    input_for_chain['inp_question'] = reader_dict['question_txt']
    input_for_chain['candidate_answer'] = reader_dict['tapex_answer_text']
    hits = reader_dict['top100_table_uid'][0:topk]

    for count,hit in enumerate(hits):
        tb = "/data/ott-qa/new_csv/" + hit + ".csv"
        #tb.to_csv('/data/ott-qa/csv/temp/tab1.csv',index=False,sep='#')
        #loader = CSVLoader(file_path='/data/ott-qa/csv/temp/tab1.csv',
        #                csv_args={"delimiter": "#"})
        loader = CSVLoader(file_path=tb, csv_args={"delimiter": ","})
        try:
            doc = loader.load()
        except:
            continue
        doc_n = f"doc{count+1}"
        input_for_chain[doc_n] = doc


        score_n = f"score{count+1}"
        input_for_chain[score_n] = reader_dict['top100_table_score'][count]

    return(input_for_chain)


#--------------- funcoes (início) ------------------

def preprocess_tables(tables: list):
    processed = []
    # loop through all tables
    for table in tables:
        # convert the table to csv and
        processed_table = "\n".join([table.to_csv(index=False)])
        # add the processed table to processed list
        processed.append(processed_table)
    return processed




def build_template_rag(num_tables):

    docs = ''
    for count in range(num_tables):
        doc = f"doc{count+1}"
        doc = "{" + doc + "},"
        docs = docs + doc
    docs = docs[0:-1]

    scores = ''
    for count in range(num_tables):
        score = f"score{count+1}"
        score = "{" + score + "},"
        scores = scores + score
    scores = scores[0:-1]


    question = "{inp_question}"
    prompt_template = f"""You are a very smart oil and gas engineer working at Petrobras in Brazil.\
            Your task is generate a technical and comprehensive answer to questions that is delimited by triple backticks \
            
            The answer must be based on the tables you will receive that are delimited by triple backticks. \
            
            First you have to translate the question to english <<<question_english>>> and answer the question in english. \
            
            Each table has a score showing how relevant the table is about the question. \
            To show how relevant the table is about the question, you will receive tables scores that is delimited by triple backticks \
            For each table check whether it is related to the question. \

            Only use tables that are related to the question to answer it.  \
            Ignore tables that are not related to the question. \
            If the answer exists in several tables, summarize them. \

            Always use references in the form [NUMBER OF DOCUMENT] when using information from a document. e.g. [3], for Document[3], \
            and write the document file name in <<<source>>>. e.g. taba.html.csv, for Document taba.html.csv
            The reference must only refer to the number that comes in square brackets after passage. \
            Otherwise, do not use brackets in your answer and reference ONLY the number of the passage without mentioning the word passage. \

            If the tables can't answer the question or you are unsure say: 'The answer can't be found in the text'. \

            You will try your best to write a concise answer in a didactic yet detailed way, \
            being truthful to the original tables. \

            Do not add any other information besides the <<<question>>>, <<<question_english>>>, <<<table_name>>>, <<<file_name>>> and <<<answer>>>.  \
            Only answer based on the tables provided. Don't make things up. \

            Format the output as JSON with the following keys: \
            question: <<<question>>> \
            question_english: <<<question_english>>> \
            answer:    <<<answer>>>  \
            source:    <<<source>>>  \

            These are the question, the tables and the scores: \
            question: ```{question}``` \
            tables:   ```{docs}``` \
            scores:   ```{scores}``` \

            """

    return(prompt_template)

def build_template_final_answer(num_tables):

    docs = ''
    for count in range(num_tables):
        doc = f"doc{count+1}"
        doc = "{" + doc + "},"
        docs = docs + doc
    docs = docs[0:-1]

    scores = ''
    for count in range(num_tables):
        score = f"score{count+1}"
        score = "{" + score + "},"
        scores = scores + score
    scores = scores[0:-1]


    question = "{inp_question}"
    candidate_answer  = "{candidate_answer}"


    #You are a very smart oil and gas engineer working at Petrobras in Brazil.\
    # generate a technical and comprehensive answer
    

    prompt_template = f"""
            Your task is generate a comprehensive answer to questions that is delimited by triple backticks. \
            The answer must be based on the tables you will receive that are delimited by triple backticks and \
            on a candidate answer generated by an informational retriever system that are delimited by triple backticks.\
            You will do it in multiple steps:

            1. translate the question to english <<<question_english>>> to answer the question in english. \
            2. Verify how relevant each table is about the question. \
            For this, you will receive table scores that are delimited by triple backticks. \
            A score closes to 1 shows that the table has high relevance to the question. \
            A score closes to 0 shows that the table has low relevance to thee question. \
            
            3. Select tables that are related to the question to answer it.  \
            Ignore tables that are not related to the question. \
            If the answer exists in several tables, summarize them. \

            4. Try your best to write a concise answer in a didactic yet detailed way, being truthful to the original tables. \

            5. Make references in the form [NUMBER OF DOCUMENT] when using information from a document. e.g. [3], for Document[3], \
            and write the document file name in <<<source>>>. e.g. taba.html.csv, for Document taba.html.csv
            The reference must only refer to the number that comes in square brackets after passage. \
            Otherwise, do not use brackets in your answer and reference ONLY the number of the passage without mentioning the word passage. \


            6. If the tables can't answer the question or you are unsure ,\
            read the candidate answer and if it is a a suitable answer to the question, you may use it.\
            if it´s not suitable, don´t use it and say: 'The answer can't be found in the tables and neither by the reader'. \
            If the candidate answer was used, the <<<reader>>> must be set to True, otherwise set to False

            7. Finally, do not add any other information besides the <<<question>>>, <<<question_english>>>, <<<table_name>>>, <<<file_name>>> and <<<answer>>> \
            and only answer based on the tables provided. Don't make things up. \

            Let's think step by step.

            Format the output as JSON with the following keys: \
            question: <<<question>>> \
            question_english: <<<question_english>>> \
            answer:    <<<answer>>>  \
            source:    <<<source>>>  \
            reader:    <<<reader>>>  \

            These are the question, the tables and the scores: \
            question: ```{question}``` \
            tables:   ```{docs}``` \
            scores:   ```{scores}``` \
            candidate answer: ```{candidate_answer}```
            """

    return(prompt_template)
