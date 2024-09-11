# usa a saída do rerank e aplica o llm para relevancia das colunas
# na saída do retriever e não do reranking
# isto é para fazer o baseline e pq o rerank nao melhorou quase nada
  
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
import utils
# retrieve
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


def build_template_new(num_tables):  # num_tables hoje é desnecessário

    question = "{inp_question}"
    table = "{table}"
    num_columns = "{num_columns}"
    prompt_template = f"""
            ## ROLE
            You are a knowledge worker very good in question and answer systems.

            ## TASK
            Your task is to identify all columns in the provided <<<table>>> and determine their relevance \
                for extracting the answer to the given <<<question>>> in a question-and-answer system.
            For each column, calculate a relevance <<<score>>> on a scale from 0 to 1, indicating how relevant \
                the column is to the provided <<<question>>>.
            
            Create two lists:
            - <<<column_name_list>>>: A list containing the valid column names.
            - <<<column_score_list>>>: A list containing the relevance scores.
            
            Important:
            - You **must not** include any column named `"page_content"` or `"metadata"` in the final output.
            - Ensure that the final lists include all columns from the <<<table>>> except `"page_content"` and `"metadata"`.
            - The number of columns in the table is provided as <<<num_columns>>>, so your output lists must contain exactly <<<num_columns>>> valid columns.
            - If the count of identified columns does not match <<<num_columns>>>, recheck and ensure all columns are included (except `"page_content"` and `"metadata"`).

            You must not fabricate column names or skip any valid columns. The final lists must match the provided number of columns.

            ## INPUTS
            question: ```{question}```\
            table:   ```{table}```\
            num_columns: ```{num_columns}```\

            ## CONSTRAINTS
            - Only include information related to the <<<question>>>, <<<column_name_list>>>, and <<<column_score_list>>> in your response.
            - If any column is named "page_content" or "metadata," **automatically exclude** it from the final lists, but ensure the output lists match the total number of valid columns, which should be exactly <<<num_columns>>>.

            ## VERIFICATION
            Before finalizing your answer, verify:
            1. That the number of columns in the final lists is exactly equal to <<<num_columns>>>.
            2. That no column named `"page_content"` or `"metadata"` is included.

            If these conditions are not met, revise the lists and check all columns again.

            ## output
            Format the output as JSON with the following keys: \
            - question: <<<question>>> \
            - column_name: <<<column_name_list>>>  \
            - score: <<<column_score_list>>>  \
            """
    return(prompt_template)



def build_template_new_old(num_tables):  # num_tables hoje é desnecessário

    question = "{inp_question}"
    table = "{table}"
    num_columns = "{num_columns}"
    prompt_template = f"""
            ## ROLE
            You are a knowledge worker very good in question and answer systems.

            ## TASK
            Your task is identify columns in the table that are not relevant \
                for extracting the answer for the provided question in a question and answer system. \
            For this, you must calculate for each column the relevance <<<score>>>, in a scale form 0 to 1, indicating how relevant the column is. \
            Create two lists: \
            the list <<<column_name_list>>> with the column names and \
            the list <<<column_score_list>>> the score numbers.  \
            
            Important:
            - You **must not** include any column named `"page_content"` or `"metadata"` in the final output.
            - Ensure that the final lists include all columns from the <<<table>>> except `"page_content"` and `"metadata"`.
            - The number of columns in the table is provided as <<<num_columns>>>, so your output lists must contain exactly <<<num_columns>>> valid columns.
            - If the count of identified columns does not match <<<num_columns>>>, recheck and ensure all columns are included (except `"page_content"` and `"metadata"`).

            You can not change any column name and the <<<column_name_list>>> can contain only column name from the <<<table>>>\
            Do not fabricate information, and follow a careful step-by-step process.


            ## INPUTS
            question: ```{question}```\
            table:   ```{table}```\
            num_columns: ```{num_columns}```\

            ## CONSTRAINTS
            - Only include information related to the <<<question>>>, <<<column_name_list>>>, and <<<column_score_list>>> in your response.
            - If any column is named "page_content" or "metadata," **automatically exclude** it from the final lists, redo the task ensuring the output lists match the total number of valid columns, which should be exactly <<<num_columns>>>.

            ## VERIFICATION
            Before finalizing your answer, verify:
            1. That the number of columns in the final lists is exactly equal to <<<num_columns>>>.
            2. That no column named `"page_content"` or `"metadata"` is included.

            If these conditions are not met, revise the lists and check all columns again.

            ## output
            Format the output as JSON with the following keys: \
            - question: <<<question>>> \
            - column_name: <<<column_name_list>>>  \
            - score: <<<column_score_list>>>  \

            ## EXAMPLE
            table:
                ,Remaining Amortization period,Fair Value of Notes (Level 2)
                ,(years),(in thousands)
                2020 Notes,0.7,"$500,855"
                2021 Notes,2.0,"$806,232"
                2025 Notes,5.7,"$528,895"
                2026 Notes,6.7,"$786,915"
                2029 Notes,9.7,"$1,063,670"
                2049 Notes,29.7,"$828,188"
            question: Which year has the greatest total accumulated amortization?
            num_columns: 3
            column_name_list: ['unnamed0', 'Remaining Amortization period', 'Fair Value of Notes (Level 2)']
            column_score_list: [1, 0.8, 0.6]
            """

            ## CONSTRAINTS
            #- Only include information related to the <<<question>>>, <<<column_name_list>>>, and <<<column_score_list>>> in your response.
            #- If you encounter any column named "page_content" or "metadata," assume there is a mistake in the table and redo the task without those columns.



    return(prompt_template)
    
def build_template_old2(num_tables):

    question = "{inp_question}"
    table = "{table}"
    prompt_template = f"""
            ## ROLE
            You are a knowledge worker very good in question and answer systems.

            ## TASK
            Your task is identify the column names in the table that are not relevant \
                for extracting the answer for the provided question in a question and answer system. \
            For this, you must calculate for each column the relevance <<<score>>>, in a scale form 0 to 1, indicating how relevant the column is. \
            Create two lists: \
            the list <<<column_name_list>>> with the column names and \
            the list <<<column_score_list>>> the score numbers.  \
            You can not change any column name and the <<<column_name_list>>> can contain only column name from the <<<table>>>
            Don't make things up. 
            Let's think step by step.

            ## INPUTS
            question: ```{question}```    \
            table:   ```{table}```        \

            ## CONSTRAINTS
            Do not add any other information besides the <<<question>>>, <<<table_name>>>, <<<column_name_list>>> and <<<column_score_list>>>.
            
            ## output
            Format the output as JSON with the following keys: \
            question: <<<question>>> \
            column_name: <<<column_name_list>>>  \
            score: <<<column_score_list>>>  \
            """
                #The answer must have be like: \
                #Relevant column names and scores: <<<column name, score>>> \

    return(prompt_template)

def build_template_errors(num_tables):

    question = "{inp_question}"
    table = "{table}"
    prompt_template = f"""

    ## ROLE
    You are an expert in question-and-answer systems.

    ## TASK
    Your task is to identify the columns in the provided <<<table>>> that are not relevant for extracting \
        the answer to the given <<<question>>> in a question-and-answer system.\
        Follow these steps carefully:

        1. Count how many columns <<<cols>>> are in the provided <<<table>>>. Note that some columns may be unnamed.

        2. Calculate the relevance <<<score>>> for each column, using a scale from 0 to 1, to indicate how relevant \
        each column is to answering the provided <<<question>>>.

        3. Create two lists:\
        - <<<column_name_list>>>: A list of the column names.\
        - <<<column_score_list>>>: A list of the corresponding relevance scores.\

        Ensure that both lists are of equal size, with <<<cols>>> entries.

        Please do not fabricate any information. Think step by step.

    ## INPUTS
    question: ```{question}```    
    table:   ```{table}```        

    ## CONSTRAINTS
    - Only include information related to the <<<question>>>, <<<column_name_list>>>, and <<<column_score_list>>> in your response.
    - If any column is named "page_content" or "metadata," there is an error, and you should repeat the task.

    ## OUTPUT
    Format the output as JSON with the following keys:
    - question: <<<question>>> 
    - column_names: <<<column_name_list>>>  
    - scores: <<<column_score_list>>>  

    ## EXAMPLE
    <<<table>>>
    ,Remaining Amortization period,Fair Value of Notes (Level 2)
    ,(years),(in thousands)
    2020 Notes,0.7,"$500,855"
    2021 Notes,2.0,"$806,232"
    2025 Notes,5.7,"$528,895"
    2026 Notes,6.7,"$786,915"
    2029 Notes,9.7,"$1,063,670"
    2049 Notes,29.7,"$828,188"

    <<<question>>>
    Which year has the greatest total accumulated amortization?

    <<<cols>>>: 3
    <<<column_name_list>>>: ['unnamed0', 'Remaining Amortization period', 'Fair Value of Notes (Level 2)']
    <<<column_score_list>>>: [1, 0.8, 0.6]
    """



    return(prompt_template)


def build_template_old(num_tables):

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




def build_input(inp_question, table_id, num_columns):
    input_for_chain = {}
    #file_path= str(table_id) # '/data/ott-qa/new_csv/' + table_id + '.csv'

    ## diminuindo a tabela em 10 linhas
#    df = pd.read_csv(table_id)
    ## Extrair as primeiras 10 linhas

    ## tabela inteira
#    first_five_lines = df.head(10)
    ## Gravar as primeiras 10 linhas em um novo arquivo CSV
#    table_id = table_id.replace('/csv/','/new_csv_10linhas/')


#    first_five_lines.to_csv(table_id, index=False)


    df_temp = pd.read_csv(table_id)
    df_temp.columns

    loader = CSVLoader(file_path=table_id,
                        metadata_columns=df_temp.columns,
                        csv_args={"delimiter": ","})
    table = loader.load()          

    input_for_chain['inp_question'] = inp_question
    input_for_chain['table'] = table
    input_for_chain['num_columns'] = num_columns
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


def build_llm_table_opt(retriever_file, device, llm):


    with open('/QA/Bert/code/path_to_files.json', 'r') as file:
        path_to_files = json.load(file)


    num_candidates = 10  #0
            # fazer aqui a leitura de todas as perguntas e a resposta

    dataset = pd.read_csv(retriever_file, sep=',') #, converters={'top100_table_uid': converter_lista})
    #dataset = dataset.drop(dataset.index[2301:])  ### removendo as ja feitas  nao zerei o index
    dataset = dataset.drop(dataset.index[:1800])   ### aqui

    print(dataset.shape)
 

    df = pd.DataFrame()


    for index, data in dataset.iterrows():
            inp_question   = data['question_txt']
                # pegando a saída do retriever ou reranking, isto é o primeiro da top100 dele
            new_data = {}
            new_data['question_idx'] = data.question_idx
            new_data['question_id']  = data.question_id
            new_data['question'] = inp_question
            new_data['model'] = retriever_file
            new_data['prompt_version'] = 'new-important'
            
            df_per_question = pd.DataFrame()
            for idx in range(num_candidates):  # fazendo para cada uma das topn
                    error = False
                    pos = 'top' + str(idx+1) # de top1 ate top10
                    new_data['position'] = pos

                    #atencao, fazendo para a saida do reranker e nao do retriever
                    answer_table_uid = eval(data['rr_top100_table_uid'])[idx].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.').replace('*', '')  
# aqui versao ott-qa  table_uid = "/data/ott-qa/new_csv/" + answer_table_uid + ".csv"
                    table_uid = answer_table_uid
                    new_data['table_uid'] = answer_table_uid
                    df_tt = pd.read_csv(table_uid)
                    num_columns = len(df_tt.columns)

                    prompt_template = build_template_new(num_candidates)
                    prompt =  ChatPromptTemplate.from_template(prompt_template)
                    chain = LLMChain(llm=llm, prompt=prompt,verbose=False)
                    status = 'OK'
                    try:    # problema no LLM, exemplo violar politica PB 
                                    input_data_dict = build_input(inp_question, table_uid, num_columns)
                                    resposta = chain.invoke(input_data_dict)
                    except:
                                    error = True
                                    status = 'Erro no llm'

                    try:    # problema na saída do LLM, exemplo trocar as chaves do json de saída
                                    output_dict = json.loads(resposta['text'])
                                    new_data['column_names']  = output_dict['column_name']
                                    new_data['column_scores'] = output_dict['score']
                    except:
                                    error = True
                                    status = 'Erro na geração'

                    if error == True:
                            new_data['column_names'] = ['error']
                            new_data['column_scores'] = [0]

                    new_data['status'] = status
                    new_data['num_columns'] = num_columns 

                    if not num_columns == len(output_dict['score']):
                        new_data['verify'] = 'mismatch'
                    else:
                        new_data['verify'] = 'ok'

                    print(index)
                    print(f'status: {status}')
                    print('table: ', new_data["table_uid"])
                    print('question: ', new_data['question'])
                    print(new_data['column_names'])
                    print(new_data['column_scores'])
                    print(new_data['num_columns'])
                    print(new_data['verify'])
                    tempo_aleatorio = random.randint(1, 2)
                    tempo_aleatorio = 1
                    print(f"Aguardando por {tempo_aleatorio} segundos...")
                    time.sleep(tempo_aleatorio)
                    df_per_question = pd.concat([df_per_question, pd.DataFrame([new_data])], ignore_index=True)

            df = pd.concat([df, df_per_question], ignore_index=True)


            if index % 50 == 0:

                llm_table_opt_file = path_to_files['llm_table_opt_destination'] + retriever_file.split('/')[-1]
                llm_table_opt_file = llm_table_opt_file.replace('csv',f'{index}.csv')


                #llm_opt_file = retriever_file.replace('/retriever/','/llm_table_opt/')
                df.to_csv(llm_table_opt_file, sep=',', index=False)       
                print(f'criado {llm_table_opt_file}')
                          
    return(df)


 

def main():

    # parametros de configuracao
    with open('/QA/Bert/code/path_to_files.json', 'r') as file:
        path_to_files = json.load(file)

    run_llm_table_opt = True
    if run_llm_table_opt == True:
        retriever_output = ['mpnet_table_intro_embeddings_cpu_512_512.csv']
                            #'mpnet_table_embeddings_cpu_512_512.csv']
                            #'mpnet_table_intro_embeddings_cpu_384_512.csv',
                            #'mpnet_table_embeddings_cpu_384_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_512_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_384_512.csv']

        for retriever_file in retriever_output:
            retriever_file = path_to_files['llm_table_opt_source'] + retriever_file
            device = 'CPU'
            #retriever_file =  f'/data/ott-qa/retriever/{retriever_file}'
            device = 'CPU'
            df_llm_opt = llmtable.build_llm_table_opt(retriever_file,"cpu",llm=llm)
            
            llm_opt_file = path_to_files['llm_table_opt_destination'] + retriever_file.split('/')[-1]
            df_llm_opt.to_csv(llm_opt_file, sep=',', index=False)       
            print(f'criado {llm_opt_file}')
            print("")



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
 
if __name__ == '__MMMMmain__':
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


def reduz_list_100_10(lst):
    lst = eval(lst)
    return lst[:10]


def concatena_tat_files():

    #OK!!!
    #file1 = '/data/tat-qa/llm_table_opt/fromrerankingcenario4/mpnet_table_intro_embeddings_cpu_512_512_filtered.de0a500.csv'
    #file1 = '/data/tat-qa/llm_table_opt/fromreranking/mpnet_table_intro_embeddings_cpu_512_512_filtered.de0a450.csv'
    file1 = '/data/tat-qa/llm_table_opt/fromrerankingcenario4/mpnet_table_intro_embeddings_cpu_512_512_filtered.de0a500.csv'
    df_file1 = pd.read_csv(file1, sep=',')
    df_file1 = df_file1.drop(df_file1.index[-10:])

    print(df_file1.shape)
    #df_file1['top100_table_intro'] = df_file1['top100_table_intro'].apply(reduz_list_100_10)

    #file2 = '/data/tat-qa/llm_table_opt/fromrerankingcenario4/mpnet_table_intro_embeddings_cpu_512_512_filtered.de500a1100.csv'
    #file2 = '/data/tat-qa/llm_table_opt/fromreranking/mpnet_table_intro_embeddings_cpu_512_512_filtered.de450.ao.final.csv'
    file2 = '/data/tat-qa/llm_table_opt/fromrerankingcenario4/mpnet_table_intro_embeddings_cpu_512_512_filtered.de500a1800.csv'

    df_file2 = pd.read_csv(file2, sep=',')
    df_file2 = df_file2.drop(df_file2.index[-10:])
    print(df_file2.shape)

    #OK!!!
    #file3 = '/data/tat-qa/llm_table_opt/fromrerankingcenario4/mpnet_table_intro_embeddings_cpu_512_512_filtered.de1100aofinal.csv'
    file3 = '/data/tat-qa/llm_table_opt/fromrerankingcenario4/mpnet_table_intro_embeddings_cpu_512_512_filtered.de1800aofinal.csv'

    df_file3 = pd.read_csv(file3, sep=',')
    #df_file3 = df_file3.drop(df_file3.index[:10])
    print(df_file3.shape)


    df = pd.concat([df_file1.reset_index(drop=True), df_file2.reset_index(drop=True) , df_file3.reset_index(drop=True)], axis=0)

    print(df.shape)

    ## adicionando o question id
    #questions_file = '/data/tat-qa/reranking/intro_improved_filter/mpnet_table_intro_embeddings_cpu_512_512_filtered.csv'

    #dataset = pd.read_csv(questions_file, sep=',')
    #question_id_list = dataset.question_id
    #print(len(question_id_list))
    #nova_lista = []
    #for elemento in question_id_list:
    #    nova_lista.extend([elemento] * 10)

    #print(len(nova_lista))

    #df.insert(0, 'question_id', nova_lista)
    #print(df.shape)

    #table_opt_file = '/data/tat-qa/llm_table_opt/fromreranking/mpnet_table_intro_embeddings_cpu_512_512_filtered.full.csv'
    table_opt_file = '/data/tat-qa/llm_table_opt/fromrerankingcenario4/mpnet_table_intro_embeddings_cpu_512_512_filtered.full.csv'
    df.to_csv(table_opt_file, sep=',', index=False)

def build_dataset():
    dataset_file = 'tat-qa'
    with open(f'/data/{dataset_file}/released_data/dev.json', 'r') as f:    # perguntas e respostas
        data = json.load(f)
        dataset = utils.convert_dataset(data)

    table_opt_file = '/data/tat-qa/released_data/dev.csv'
    dataset.to_csv(table_opt_file, sep=',', index=False)


def remove_questions(file_source, to_remove):
    df = pd.read_csv(file_source, sep=',')
    df_to_remove = pd.read_csv(to_remove, sep=',')
    list_to_remove = df_to_remove['question_id'].to_list()

    print(df.shape)
    df_filtered = df[~df['question_id'].isin(list_to_remove)]
    print(df_filtered.shape)
    new_file = file_source.replace(".csv","_filtered.csv")
    df_filtered.to_csv(new_file, sep=',', index=False)


def list_error_to_remove():
    # cria o csv remover com as questions em top1,2,3 que deram erro no llm
    # nao estou usando nesse momento
    fin = '/data/tat-qa/llm_table_opt/mpnet_table_intro_embeddings_cpu_512_512.csv'
    df = pd.read_csv(fin, sep=',')
    print(df.shape)
    df_remover = df[(df['position'] == 'top1') | (df['position'] == 'top2') | (df['position'] == 'top3')]
    df_remover = df_remover[df_remover['status'] != 'OK']
    list_remover = df_remover.question_id.to_list()
    list_remover = list(set(list_remover))
    df_out = df = pd.DataFrame(list_remover, columns=['question_id'])
    print(df_out.shape)
    new_file = '/data/tat-qa/llm_table_opt/remover.2024.08.10.csv'
    df_out.to_csv(new_file, sep=',', index=False)
    return


def verifica_coluna_unnamed():
    fin = '/data/tat-qa/llm_table_opt/mpnet_table_intro_embeddings_cpu_512_512.csv'
    df = pd.read_csv(fin, sep=',')
    print(df.shape)

    # Criando a nova coluna usando a função any() e a expressão regular
    df['contains_unnamed0'] = df['column_names'].apply(lambda x: 'Unnamed: 0' in x)
    df = df[df['position'] == 'top1']
    print(df['contains_unnamed0'].sum())
    df = df[df['contains_unnamed0'] == True]
    
    list_remover = df.question_id.to_list()
    list_remover = list(set(list_remover))
    df_out = df = pd.DataFrame(list_remover, columns=['question_id'])
    print(df_out.shape)
    new_file = '/data/tat-qa/llm_table_opt/remover.unnmamed0.2024.08.10.csv'
    df_out.to_csv(new_file, sep=',', index=False)
    return



#verifica_coluna_unnamed()
#exit()
#list_error_to_remove()

#fin = '/data/tat-qa/llm_table_opt/mpnet_table_intro_embeddings_cpu_512_512.csv'
#to_remove = '/data/tat-qa/llm_table_opt/remover.unnmamed0.2024.08.10.csv'
#remove_questions(fin, to_remove)

 
#concatena_tat_files()
#print()

#exit()

#exit()


#question_to_remove = df_remover.question_id
#question_to_remove = list(set(question_to_remove))

#df
#print(len(question_to_remove))
#df = df[df['status'] == 'OK']
#df = df[df['position'] == 'top1']
#print(df.shape)


