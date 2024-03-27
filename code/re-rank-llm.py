#from tabfact import get_embeddings
#quais são as atividades que são medidas em pes cubicos?
# qual é a atividade que dá maior produção de barris por dia?
#qual é Capacidade de operação de refino?
#quem é Obama?
#what nationality is mack kalvin?
#who is john dramani mahama?


import torch
import os, json
import pickle
import time, random
import pandas as pd
# retriever
from sentence_transformers import SentenceTransformer, util


#reader
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering
from transformers import TapasConfig, AutoConfig


# LLM generator
from langchain.chat_models import AzureChatOpenAI
from openai.api_resources.abstract import APIResource
from langchain.document_loaders import CSVLoader

# LLM Chain
#from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory

# parser output
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

#apid.petrobras.com.br
#APID:6a1b62decb914f4291b754bea0f92d21
#apit.petrobras.com.br
#APIT:239042549f214f75abc4a006034eb4d0
#api.petrobras.com.br
#API:72b26ee264b5440ca36cdf717ee80712

os.environ["OPENAI_API_KEY"] = '72b26ee264b5440ca36cdf717ee80712'
os.environ["OPENAI_API_BASE"] = 'https://api.petrobras.com.br'
os.environ["OPENAI_API_VERSION"] = '2023-03-15-preview'
os.environ["OPENAI_API_TYPE"] = 'azure'
os.environ["REQUESTS_CA_BUNDLE"] = "/nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/petrobras-openai/petrobras-ca-root.pem"
APIResource.azure_api_prefix = 'ia/openai/v1/openai-azure/openai'
print(os.environ["REQUESTS_CA_BUNDLE"])

import warnings
warnings.filterwarnings("ignore")
####################################funcoes###########################


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


def get_embeddings(model_name, path, max_corpus_size):
    # Some local file to cache computed embeddings

    start_time = time.time()
    embedding_file = f'mpnet_table_embeddings_{device}_384_514.pkl'

    embedding_file = path + embedding_file
    # Check if embedding cache path exists
    if not os.path.exists(embedding_file):
        print(f'Embeddings nao encontrado em cache {embedding_file}')
        print('abortando')
        exit()
    print(f'Embeddings cache {embedding_file} encontrado localmente...')
    print("Aguarde carregando....")

def build_template_new(num_tables):

    question = "{inp_question}"
    table = "{table}"
    prompt_template = f"""
            For the provided question and the provided table, you have to identify the column names in the table that are not relevant \
                for extracting the answer in a question and answer system. \
            For this, you must calculate for each column the relevance <<<score>>>, in a scale form 0 to 1, indicating how relevant the column is. \
            Create two lists: \
            the list <<<column_name_list>>> with the column names and \
            the list <<<column_score_list>>> the score numbers.  \
            You can not change any column name and the <<<column_name_list>>> can contain only column name from the <<<table>>>

            These are the question and the table: \
            question: ```{question}```    \
            table:   ```{table}```        \

            Do not add any other information besides the <<<question>>>, <<<table_name>>>, <<<column_name_list>>> and <<<column_score_list>>>.
            Format the output as JSON with the following keys: \
            question: <<<question>>> \
            column_name:    <<<column_name_list>>>  \
            score:    <<<column_score_list>>>  \

            """
                #The answer must have be like: \
                #Relevant column names and scores: <<<column name, score>>> \

    return(prompt_template)


#----------------------
#            The answer must have two parts. \
#            The firt part has to be like: \
#                Relevant tables: <<<table_name>>> \
#            the second part has to be like: \
#                    -------------------- \
#                    Question: <<<question>>>\
#                    Question in english: <<<question_english>>>
#                    Answer: <<<answer>>> \
#            Do not add any other information besides the <<<question>>>, <<<table_name>>>, <<<answer>>>.  \
#
#            question: ```{question}```


#---------------




def build_input(inp_question, table_id):
    input_for_chain = {}
    input_for_chain['inp_question'] = inp_question
    file_path='/data/ott-qa/new_csv/' + table_id + '.csv'
    loader = CSVLoader(file_path=file_path,
                        csv_args={"delimiter": ","})
    table = loader.load()          
    input_for_chain['table'] = table

    return(input_for_chain)

#{'inp_question':inp_question,'doc1':doc1,'doc2':doc2,'doc3':doc3}

#######################################################################

def get_rr_output(file, device):
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
                    'llm_ret_column_names'   : [],   # nome colunas depois de aplicar llm na saída do retriever
                    'llm_ret_column_scores'  : [],   # scores das colunas depois de aplicar llm na saída do retriever
                    'llm_ret_column_counts'  : [],   # novo numero de colunas depois de aplicas llm ""       ""
                    'top_column_names'       : [],   # colunas originais
                    'top_column_counts'      : [],   # num colunas originais

                    }
        predictions.append(new_data)
    return(predictions)

def get_top100_tables(retriever_full_list, list_top100_id):
    tables_top100_list = []
    for table_id in list_top100_id:
        #table_id = retriever_full_list[idx]['table_id']
        file_name = f'/data/ott-qa/new_csv/{table_id}.csv'
        df = pd.read_csv(file_name, sep=',')
        tables_top100_list.append(df)
    return tables_top100_list



def main():

    retriever_output_files =  [#'mpnet_table_cpu_512_514.json']                   # só a tabela
                               #'mpnet_table_header_cpu_512_514.json',            # tabela mais a introdução do documento]
                               #'mpnet_table_intro_cpu_512_514.json', #,             # tabela mais a o header da tabela
                               #'mpnet_table_section_title_cpu_512_514.json',     # tabela mais uma passage da seção da tebale
                               #'mpnet_table_section_text_cpu_512_514.json'] #,      # tabela mais o titulo da tabela
                               #'mpnet_table_cpu_384_514.json',    ## esses mais antigos nao tem o tables_and_append_tokens_len
                               #'mpnet_table_header_cpu_384_514.json'] #,
                               'mpnet_RR_table_cpu_384_514.json']
                               #'mpnet_RR_table_header_cpu_384_514.json']
                               #'mpnet_RR_table_intro_cpu_384_514.json',
                               #'mpnet_RR_table_section_title_cpu_384_514.json',
                               #'mpnet_RR_table_section_text_cpu_384_514.json']


    num_candidates = 10  #0
    # fazer aqui a leitura de todas as perguntas e a resposta



    for retriever_output_file in retriever_output_files:  # para cada retriever feito
        rr_output_list = []
        retriever_dict_list = get_rr_output(retriever_output_file,device)

        count = 1
        reranking_LLM_list = []
        print(f'Processando {retriever_output_file}')

        reranking_LLM = {}
        count_special = 201

        for retriever_output in retriever_dict_list[200:250]:  ### atencao aqui
            reranking_LLM = retriever_output.copy()
            inp_question = retriever_output['question']
            #top100_tables = get_top100_tables(retriever_dict_list, retriever_output['top_uid'])

            #table_uid_column_names = []
            #table_uid_scores = []
            #table_uid_ret_column_names = []    
            #table_uid_ret_column_scores = []
            #table_uid_ret_column_counts = []

            #top_column_names_list = []   # colunas originais
            #top_column_count_list = []  # num colunas originais

            #llm_ret_column_names_list = []   # nome colunas depois de aplicar llm na saída do retriever
            #llm_ret_column_scores_list = []  # score colunas depois de aplicar llm na saída do retriever
            #llm_ret_column_counts_list = []  # nume colunas depois de aplicar llm na saída do retriever 


            top_column_counts = []
            top_column_names = []
            llm_ret_column_names  = [] # nome colunas de uma tabela
            llm_ret_column_scores = [] # scores das colunas de uma tabela
            llm_ret_column_counts = [] # num colunas de uma tabela

            count_top = 0  # fazendo somente os top10
            for table_id in retriever_output['top_uid']:  # fazendo na saida do retriever

                #processa_LLM()
                print(inp_question)
                prompt_template = build_template_new(num_candidates)
                prompt =  ChatPromptTemplate.from_template(prompt_template)
            
                #chain = LLMChain(llm=llm, prompt=prompt,verbose=False, memory=memory)
                chain = LLMChain(llm=llm, prompt=prompt,verbose=False)
                #df_table = pd.read_csv(file_name, sep=',')

                start_time = time.time()

                try:
                    input_data_dict = build_input(inp_question, table_id)
                    resposta = chain.invoke(input_data_dict)
                except:
                    top_column_counts.append(-99)
                    top_column_names.append("error")
                    llm_ret_column_counts.append(-99)
                    llm_ret_column_names.append("error")
                    llm_ret_column_scores.append(-99)
                    count_top +=1
                    print("erro")
                    continue

                print(resposta['text'])
                print(f'table id {table_id}')
                print(count_special)
                print("")
                output_dict = json.loads(resposta['text'])

                tempo_aleatorio = random.randint(10, 15)
                print(f"Aguardando por {tempo_aleatorio} segundos...")
                time.sleep(tempo_aleatorio)


                try:
                    llm_ret_column_names.append(output_dict['column_name'])
                    llm_ret_column_scores.append(output_dict['score'])
                except:
                    top_column_counts.append(-99)
                    top_column_names.append("error")
                    llm_ret_column_counts.append(-99)
                    llm_ret_column_names.append("error")
                    llm_ret_column_scores.append(-99)
                    count_top +=1
                    continue
                
                df = pd.read_csv("/data/ott-qa/new_csv/" + table_id + ".csv")
                top_column_counts.append(df.shape[1])
                top_column_names.append(df.columns.to_list())


                # removendo as colunas com relevancia <= 0.2
                idx = 0
                for column in output_dict['column_name']:
                    try:
                        if output_dict['score'][idx] <= 0.2:
                            df.drop(column, inplace=True,axis=1)
                    except:
                        continue
                    idx +=1
                df.to_csv("/data/ott-qa/output/mpnet_RR_table_cpu_384_514/llm_ret/" + table_id + ".csv")

                llm_ret_column_counts.append(df.shape[1])


                if count_top == 9:  # fazendo somente para os top10
                    break
                count_top +=1

            reranking_LLM['llm_ret_column_names']  = llm_ret_column_names
            reranking_LLM['llm_ret_column_scores'] = llm_ret_column_scores
            reranking_LLM['llm_ret_column_counts'] = llm_ret_column_counts
            reranking_LLM['top_column_counts']     = top_column_counts
            reranking_LLM['top_column_names']      = top_column_names
            reranking_LLM_list.append(reranking_LLM)








            #if ((True)): #count_special % 25) == 0):  # fazendo somente para os top10
            #    out_put_file = "/data/ott-qa/output/ColRel/" + table_id + ".json" ##retriever_output_file
                #out_put_file = out_put_file.replace("mpnet_RR","0-500-mpnet_RR_RetriverColRel")
                #out_put_file = out_put_file.replace("RetriverColRel",f'RetriverColRel{str(count_special)}')
                
            #    with open(out_put_file, "w") as arquivo_json:
            #        json.dump(reranking_LLM, arquivo_json)
            #    print(f'criado {out_put_file}')
            count_special +=1

            #CUIDADO
        out_put_file = f'/data/ott-qa/output/mpnet_RR_table_cpu_384_514/200_250_{retriever_output_file}'
        #out_put_file = out_put_file.replace("mpnet_RR","0-500-mpnet_RR_RetriverColRel")
        with open(out_put_file, "w") as arquivo_json:
            json.dump(reranking_LLM_list, arquivo_json)
        print(f'criado {out_put_file}')

    exit()

    return


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

inp_question = ''
num_candidates = 3

debug_mode = True
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
        embedding_path = "/QA/Bert/data/ott-qa/"
        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        retriever_biencoder_model = SentenceTransformer(retriever_model_name, device=device)  # para fazer o download
    
        #reader
        reader_model_name = "google/tapas-large-finetuned-wikisql-supervised"
        config = TapasConfig(reader_model_name)    # baixando o modelo do hugging face
        reader_model = TapasForQuestionAnswering.from_pretrained(reader_model_name)
        reader_tokenizer = TapasTokenizer.from_pretrained(reader_model_name)
    
    else:
        #retriever
        embedding_path = "/data/ott-qa/embeddings/"
        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        path_local = "/modelos/deepset_all-mpnet-base-v2-table" #deepset_all-mpnet-base-v2-tablel.pth"
        retriever_biencoder_model = SentenceTransformer(path_local, device=device)
    
        # Reader
        reader_model_name = "google/tapas-large-finetuned-wikisql-supervised"
        config = TapasConfig(reader_model_name)
        path_local = "/modelos/google_model_tapas-large-finetuned-wikisql-supervised"     # pegando o modelo local
        reader_model = TapasForQuestionAnswering.from_pretrained(path_local, local_files_only=True)
        path_local = "/modelos/google_tokenizer_tapas-large-finetuned-wikisql-supervised"
        reader_tokenizer = TapasTokenizer.from_pretrained(path_local, local_files_only=True)

    # LLM Model
    llm = AzureChatOpenAI(
        deployment_name="gpt-35-turbo-16k-petrobras",
        model_name="gpt-3.5-turbo-16k",
        temperature=0,
    )
    #if  ConversationBuffer == False:
    #    memory = ConversationBufferWindowMemory(k=1)
    #else:
    #    memory = ConversationBufferWindowMemory()

    main()







