import torch
import os, json
import pickle
import time
# retriever
from sentence_transformers import SentenceTransformer, util

def get_embeddings(file, device):
    # Some local file to cache computed embeddings

    start_time = time.time()
    embedding_file = f'/data/ott-qa/embeddings/{file}'
    # Check if embedding cache path exists
    if not os.path.exists(embedding_file):
        print(f'Embeddings nao encontrado em cache {file}')
        print('abortando')
        exit()
    print(f'Embeddings cache {embedding_file} encontrado localmente...')
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
    print(f'Table + sentences embeddings load took {(time.time() - start_time)} seconds')



    return embeddings_dict


def cosine_similarity(inp_question,corpus_embeddings,num_candidates):

    start_time = time.time()
    question_embedding = retriever_biencoder_model.encode(inp_question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=num_candidates)
    hits = hits[0]  # Get the hits for the first query
    #print("")
    #print('---------------------------------------------------------------------------')
    #print(f'Cosine-Similarity para top-{num_candidates} levou {(time.time() - start_time)} seconds')
    #print(f'imprimindo Top 5 dos {num_candidates} hits with cosine-similarity:')
    #for hit in hits[0:5]:
    #    print("\t{:.3f}\t{}".format(hit["score"], corpus_tables[hit["corpus_id"]]))

    return hits


def main():

    embeddings_file =  [#'mpnet_table_embeddings_cpu_512_514.pkl',                   # só a tabela
                        'mpnet_table_header_embeddings_cpu_512_514.pkl',           # tabela mais a introdução do documento]
                        'mpnet_table_intro_embeddings_cpu_512_514.pkl',          # tabela mais a o header da tabela
                        'mpnet_table_section_title_embeddings_cpu_512_514.pkl',    # tabela mais uma passage da seção da tebale
                        'mpnet_table_section_text_embeddings_cpu_512_514.pkl',   # tabela mais o titulo da tabela
                        #'mpnet_table_embeddings_cpu_384_514.pkl'] #,    ## esses mais antigos nao tem o tables_and_append_tokens_len
                        'mpnet_table_header_embeddings_cpu_384_514.pkl',
                        'mpnet_table_intro_embeddings_cpu_384_514.pkl',
                        'mpnet_table_section_title_embeddings_cpu_384_514.pkl',
                        'mpnet_table_section_text_embeddings_cpu_384_514.pkl']





    # 0 - Question
    #inp_question = input("Entre com a question: ")
    #num_candidates = int(input("Entre com numero para Top-K candidatos: "))
    num_candidates = 100
    # fazer aqui a leitura de todas as perguntas e a resposta

    for embedding_file in embeddings_file:  # para cada tipo de embeddings feito

        embedding_dict = get_embeddings(embedding_file,device)  # dicionario com listas de todas as info
        list_tables_uid = embedding_dict['tables_uid']
        list_tables_tokens_len = embedding_dict['tables_tokens_len']
        list_tables_and_append_tokens_len = embedding_dict['tables_and_append_tokens_len']
        
        predictions = []
        count = 0
        
        with open('/data/ott-qa/released_data/dev.json', 'r') as f:    # perguntas e respostas
            data = json.load(f)
        for qa in data:
            start_time = time.time()

            inp_question = qa['question']
            answer_table_id = qa['table_id'].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.')  # ground trhuth
            
            # 1 - calcula similaridade - so tabelas
            hits = cosine_similarity(inp_question,embedding_dict['tables_embedding'], num_candidates)

            end_time = time.time()

            tables_uid = embedding_dict['tables_uid']

            index_top_list = []
            id_top_list    = []
            score_top_list = []

            for hit in hits:
        #        print(f'id {hit["corpus_id"]} | {tables_uid[hit["corpus_id"]]} | score {hit["score"]}')
                index_top_list.append(hit["corpus_id"])
                id_top_list.append(tables_uid[hit["corpus_id"]])
                score_top_list.append(hit["score"])
                #print(f'id {hit["corpus_id"]} | | score {hit["score"]}')

            ground_thruth_idx = embedding_dict['tables_uid'].index(answer_table_id)

            table_tokens_len             = list_tables_tokens_len[ground_thruth_idx]
            table_tokens_append_len  = list_tables_and_append_tokens_len[ground_thruth_idx]

            top1 = top10 = top50 = top100 = False
            if answer_table_id == id_top_list[0]:
                top1 = True
            elif answer_table_id in id_top_list[1:11]:
                top10 = True
            elif answer_table_id in id_top_list[11:51]:
                top50 = True
            elif answer_table_id in id_top_list[51:]:
                top100 = True

            top_tokens_len = []
            top_tokens_append_len = []
            for table in id_top_list:
                indice_tabela = list_tables_uid.index(table)
                top_tokens_len.append(list_tables_tokens_len[indice_tabela])
                top_tokens_append_len.append(list_tables_and_append_tokens_len[indice_tabela])


        #    print('---------------------------------------------------------------------------')

            new_data = {}
            new_data = {'question_id'  : qa['question_id'],
                        'question'     : qa['question'],
                        'table_idx'    : ground_thruth_idx,
                        'table_tokens_len' : table_tokens_len,
                        'table_tokens_append_len': table_tokens_append_len,
                        'table_id'     : qa['table_id'],
                        'answer-text'  : qa['answer-text'],
                        'top_tokens_len' : top_tokens_len, ##tables_tokens_len[tables_uid.index(answer_table_id)],
                        'top_tokens_append_len' :top_tokens_append_len, ##tables_and_append_tokens_len[tables_uid.index(answer_table_id)],
                        'top_index': index_top_list,
                        'top_uid'  : id_top_list,
                        'top_score': score_top_list,
                        'top1'         : top1,
                        'top10'        : top10,
                        'top50'        : top50,
                        'top100'       : top100,
                        'time'         : end_time - start_time}
            predictions.append(new_data)
        #    if count == 5:
        #        break
        #    count+=1
        out_put_file = "/data/ott-qa/output/" + embedding_file.replace(".pkl",".json")
        out_put_file = out_put_file.replace("_embeddings_","_")
        with open(out_put_file, "w") as arquivo_json:
            json.dump(predictions, arquivo_json)
        
    exit()
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







