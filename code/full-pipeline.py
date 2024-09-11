### dataset ott-qa
### juntando:
#  NAO reader-tapex.py ( a entrada é a saída do retriever)
# comecando pelo inicio - retriever + dataset
#  llm-qa-reader.py
# ainda sem o rerank 

#from tabfact import get_embeddings
#quais são as atividades que são medidas em pes cubicos?
# qual é a atividade que dá maior produção de barris por dia?
#qual é Capacidade de operação de refino?
#quem é Obama?
#what nationality is mack kalvin?
#who is john dramani mahama?

#1211 otimização das questions
#How many Bangor City FC competitions have been won by the 1962-63 season opponent?
#The 1962-63 season opponent has won <<<number>>> Bangor City FC competitions.
 
import torch
import os
import pickle
import time, random
import json
import pandas as pd


import utils
import retriever
import rerankingroberta
import rerankingmarcoMiniLMintro
import rerankingmarcoMiniLM
import reader
import readertableopt
import llmtable
#import llmqareader
#import evaluate
import llmquestions

# retriever
from sentence_transformers import SentenceTransformer, util
# nesse momento pego a saída do retriever, nao uso o sentencetransformer

# reranker
from sentence_transformers import CrossEncoder


#reader TAPAS
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering
from transformers import TapasConfig, AutoConfig

#reader TAPEX
from transformers import TapexTokenizer, BartForConditionalGeneration
#AutoModelForSeq2SeqLM:
#Esta classe é uma interface genérica que pode ser usada com qualquer modelo seq2seq de linguagem.


# LLM generator
from langchain.chat_models import AzureChatOpenAI
from openai.api_resources.abstract import APIResource
#from openai.resources.abstract import APIResource
from langchain.document_loaders import CSVLoader

# LLM Chain
#from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory

# parser output
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

#DATASET = 'wikitablequestions'
DATASET = 'tat-qa'


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

import datetime

####################################funcoes###########################
 

def	open_loggging(log_file,task):
    current_datetime = datetime.datetime.now().strftime("%d%m%y %H:%M:%S")
    with open(log_file, 'a') as file:
        file.write('-------Inicio-------' + '\n')
        file.write(current_datetime + '\n')
        file.write(f'entrada {task}' + '\n')

def	close_loggging(log_file,task):
    current_datetime = datetime.datetime.now().strftime("%d%m%y %H:%M:%S")
    with open(log_file, 'a') as file:
        file.write(current_datetime + '\n')
        file.write(f'criado {task}' + '\n')
        file.write('-------FIM-------' + '\n')

def main():

    # parametros de configuracao
    with open('/QA/Bert/code/path_to_files.json', 'r') as file:
        path_to_files = json.load(file)

    #print(path_to_files)

    run_retriever                = False
    run_improve_question         = False   # llm nas perguntas
    run_reranker_roberta         = False
    run_reranker_marcomini       = False
    run_reranker_marcomini_intro = False
    run_reader_tapas_tapex       = False
    run_reader_improved_tables   = True
    run_llm_table_opt         = False
    run_llm_rag_and_wikipedia = False
    run_evaluate_answers      = False
    run_llmquestions          = False

    if run_retriever == True:    ## fazendo com os novos embeddings - nome das colunas melhorados
        embeddings = [#'mpnet_table_header_embeddings_cpu_384_512.pkl',
                    #'mpnet_table_header_embeddings_cpu_512_512.pkl']#,
                    #'mpnet_table_section_title_embeddings_cpu_512_512.pkl',
                    #'mpnet_table_section_title_embeddings_cpu_384_512.pkl',
                    'mpnet_table_intro_embeddings_cpu_512_512.pkl']
                    #'mpnet_table_embeddings_cpu_512_512.pkl']
        for embedding_file in embeddings:
            embedding_file = path_to_files['retriever_source'] + embedding_file
            device = 'CPU'
            df_retriever = retriever.build_retriever(DATASET, embedding_file, "cpu", run_improve_question, llm=llm)
            retriever_file = path_to_files['retriever_destination'] + embedding_file.split('/')[-1]
            retriever_file = retriever_file.replace('.pkl','.csv')
            #if run_improve_question == True:
            #retriever_file = retriever_file.replace('/improved/','/improved003/')
            df_retriever.to_csv(retriever_file,sep=',',index=False)

    if run_reranker_roberta == True:
        diretorio = "/modelos/reranker/cross-encoder-stsb-roberta-base"
        rerank_cross_encoder_model = CrossEncoder(diretorio)
        retriever_output = [#'mpnet_table_section_title_embeddings_cpu_512_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_384_512.csv',
                            #'mpnet_table_intro_embeddings_cpu_512_512.csv',
                            'mpnet_table_embeddings_cpu_512_512.csv']
        for retriever_file in retriever_output:
            retriever_file = path_to_files['reranker_roberta_source'] + retriever_file
            device = 'CPU'
            df_reranking = rerankingroberta.build_reranking(retriever_file, "cpu", run_improve_question)
            reranking_file = path_to_files['reranker_roberta_destination'] + retriever_file.split('/')[-1]
            df_reranking.to_csv(reranking_file,sep=',',index=False)               

    if run_reranker_marcomini == True:
        diretorio = "/modelos/reranker/cross-encoder-ms-marco-MiniLM-L-6-v2"
        cross_encoder = CrossEncoder(diretorio)
        retriever_output = [#'mpnet_table_section_title_embeddings_cpu_512_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_384_512.csv',
                            #'mpnet_table_intro_embeddings_cpu_512_512.csv',
                            'mpnet_table_intro_embeddings_cpu_512_512_filtered.csv']
        for retriever_file in retriever_output:
            retriever_file = path_to_files['reranker_marcomini_source'] + retriever_file
            device = 'CPU'
            df_reranking = rerankingmarcoMiniLM.build_reranking(retriever_file, "cpu", run_improve_question)
            reranking_file = path_to_files['reranker_marcomini_destination'] + retriever_file.split('/')[-1]
            df_reranking.to_csv(reranking_file,sep=',',index=False)

    if run_reranker_marcomini_intro == True:
        diretorio = "/modelos/reranker/cross-encoder-ms-marco-MiniLM-L-6-v2"
        cross_encoder = CrossEncoder(diretorio)
        retriever_output = [#'mpnet_table_section_title_embeddings_cpu_512_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_384_512.csv',
                            'mpnet_table_intro_embeddings_cpu_512_512_filtered.csv']
                           #'mpnet_table_embeddings_cpu_512_512.csv']
        for retriever_file in retriever_output:
            retriever_file = path_to_files['reranker_marcominiintro_source'] + retriever_file
            device = 'CPU'
            df_reranking = rerankingmarcoMiniLMintro.build_reranking(retriever_file, "cpu", run_improve_question)
            reranking_file = path_to_files['reranker_marcominiintro_destination'] + retriever_file.split('/')[-1]
            df_reranking.to_csv(reranking_file,sep=',',index=False)

    if run_reader_tapas_tapex == True:
        retriever_output = [#'mpnet_table_intro_embeddings_cpu_512_512.csv',
                            #'mpnet_table_embeddings_cpu_512_512_filtered.csv']#,
                            'mpnet_table_intro_embeddings_cpu_512_512_filtered.csv']
                            #'mpnet_table_intro_embeddings_cpu_384_512.csv',
                            #'mpnet_table_embeddings_cpu_384_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_512_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_384_512.csv']



        for retriever_file in retriever_output:
            retriever_file = path_to_files['reader_source'] + retriever_file
            #retriever_file =  f'/data/ott-qa/retriever/improved/{retriever_file}'
            device = 'CPU'
            df_reader = reader.build_reader(retriever_file, "cpu", run_improve_question)
            reader_file = path_to_files['reader_destination'] + retriever_file.split('/')[-1]
            df_reader.to_csv(reader_file, sep=',', index=False)       
            print(f'criado {reader_file}')
            print("")


    if run_reader_improved_tables == True:
        retriever_output = [#'mpnet_table_intro_embeddings_cpu_512_512.csv',
                            'mpnet_table_intro_embeddings_cpu_512_512_filtered.csv']#,
                            #'mpnet_table_intro_embeddings_cpu_384_512.csv',
                            #'mpnet_table_embeddings_cpu_384_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_512_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_384_512.csv']

        log_file = "/data/tat-qa/reader_table_opt/reader_improved_tables.log"
        for retriever_file in retriever_output:
            retriever_file = path_to_files['reader_opt_source'] + retriever_file
            open_loggging(log_file, retriever_file)

            #retriever_file =  f'/data/ott-qa/retriever/improved/{retriever_file}'
            device = 'CPU'
            df_reader = readertableopt.build_reader(retriever_file, "cpu", run_improve_question)
            reader_file = path_to_files['reader_opt_destination'] + retriever_file.split('/')[-1]
            df_reader.to_csv(reader_file, sep=',', index=False)       
            print(f'criado {reader_file}')
            print("")
  
            close_loggging(log_file, reader_file)

    # otimizacao das tabelas, fazendo um score para cada coluna de relevancia perante a question.
    #
    if run_llm_table_opt == True:
        retriever_output = ['mpnet_table_intro_embeddings_cpu_512_512_filtered.csv']
                            #'mpnet_table_embeddings_cpu_512_512.csv']
                            #'mpnet_table_intro_embeddings_cpu_384_512.csv',
                            #'mpnet_table_embeddings_cpu_384_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_512_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_384_512.csv']
        log_file = "/data/tat-qa/llm_table_opt/llm_table_opt.log"
        for retriever_file in retriever_output:
                
            retriever_file = path_to_files['llm_table_opt_source'] + retriever_file
            open_loggging(log_file, retriever_file)

            #retriever_file =  f'/data/ott-qa/retriever/{retriever_file}'
            device = 'CPU'
            df_llm_opt = llmtable.build_llm_table_opt(retriever_file,"cpu",llm=llm)
            
            llm_opt_file = path_to_files['llm_table_opt_destination'] + retriever_file.split('/')[-1]
            df_llm_opt.to_csv(llm_opt_file, sep=',', index=False)       
            print(f'criado {llm_opt_file}')
            print("")

            close_loggging(log_file, llm_opt_file)


    # usando GPT para obter a resposta de duas formas,
    # primeira tendo como contexto as tabelas selecionadas pelo reader
    # segunda tendo como contexto a wikipedia
    # No final, avaliação para identificar qual das duas respostas é a melhor
    # A entrada é a saída do reader (reader_tapas_tapex)

    if run_llm_rag_and_wikipedia == True:
        reader_output =      ['mpnet_table_intro_embeddings_cpu_512_512.csv']
                            #'mpnet_table_embeddings_cpu_512_512.csv']#,        ok, com 2253 linhas
                            #'mpnet_table_intro_embeddings_cpu_384_512.csv',
                            #'mpnet_table_embeddings_cpu_384_512.csv']


        for reader_file in reader_output:
            reader_file =  f'/data/ott-qa/reader_tapex_tapas/{reader_file}'
            device = 'CPU'
            df_llm_tapex = llmqareader.build_llm_rag_and_wikipedia(reader_file,"cpu",llm=llm)
            llm_tapex_file = reader_file.replace('/reader_tapex_tapas/','/llm_tapex/')
            df_llm_tapex.to_csv(llm_tapex_file, sep=',', index=False)       
            print(f'criado: {llm_tapex_file}')
            print("")


    if run_evaluate_answers == True:
        #llm_tapex_output = #['mpnet_table_intro_embeddings_cpu_512_512.csv']  #2214 linhas, removida linha 1311 -  ## erro!!
                            #[#'intro-512-512-teste.csv']  
                            #'mpnet_table_intro_embeddings_cpu_512_512.csv'] #,       #2215 linhas - falta rag-wiki answer a partir de 1803      
                            #'mpnet_table_intro_embeddings_cpu_384_512.csv',
        llm_tapex_output =  ['mpnet_table_intro_embeddings_cpu_512_512.csv']
 
        # faltando a linha 1311 - ver arqquivo teste
        for llm_file in llm_tapex_output:
            llm_file =  f'/data/ott-qa/llm_tapex/{llm_file}'
            device = 'CPU'
            df_evaluate = evaluate.build_evaluate_answers(llm_file,"cpu",llm=llm)
            evaluate_file = llm_file.replace('/llm_tapex/','/llm_tapex_evaluate/')
            df_evaluate.to_csv(evaluate_file, sep=',', index=False)       
            print(f'criado: {evaluate_file}')
            print("")

    if run_llmquestions == True:
        llmquestions.build_improve_questions()



    exit()

def request_llm(input_data_dict, chain):
        output_dict = {}
        try:
            resposta = chain.invoke(input_data_dict)
            output_dict = json.loads(resposta['text'])
            output_dict['status_indomain'] = 'success'
            return output_dict
        except:
            output_dict['status_indomain'] = 'failure'
            return output_dict

def request_wikipedia_llm(input_data_dict, chain):
        output_dict = {}
        try:
            resposta = chain.invoke(input_data_dict)
            output_dict = json.loads(resposta['text'])
            output_dict['status_wikipedia'] = 'success'
            return output_dict
        except:
            output_dict['status_wikipedia'] = 'failure'
            return output_dict




# https://dev.to/lgrammel/tutorial-create-an-ai-agent-that-reads-wikipedia-for-you-31cm
#You are an knowledge worker that answers questions using Wikipedia content.
## CONSTRAINTS
#All facts for your answer must be from Wikipedia articles that you have read.

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

debug_mode = False
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

    download_models = False
    local_models = True
    if download_models == True:
        #retriever
        embedding_path = "/QA/Bert/data/ott-qa/embeddings/"
        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        retriever_biencoder_model = SentenceTransformer(retriever_model_name, device=device)  # para fazer o download
    
        #reader tapas
        reader_model_name = "google/tapas-large-finetuned-wikisql-supervised"
        config = TapasConfig(reader_model_name)    # baixando o modelo do hugging face
        reader_model = TapasForQuestionAnswering.from_pretrained(reader_model_name)
        reader_tokenizer = TapasTokenizer.from_pretrained(reader_model_name)

        #reranker
        #from sentence_transformers import CrossEncoder
        #rerank_cross_encoder_model = CrossEncoder("cross-encoder/stsb-roberta-base")


    
    if local_models == True:
        #retriever - nesse momento estou usando a saída do rr
        #embedding_path = "/data/ott-qa/embeddings/"
        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        path_local = "/modelos/deepset_all-mpnet-base-v2-table" #deepset_all-mpnet-base-v2-tablel.pth"
        retriever_biencoder_model = SentenceTransformer(path_local, device=device)
        print(retriever_biencoder_model)
    
        # Reader TAPAS estava reader_model e reader_tokenizer
        reader_model_name = "google/tapas-large-finetuned-wikisql-supervised"
        config = TapasConfig(reader_model_name)
        path_local = "/modelos/google_model_tapas-large-finetuned-wikisql-supervised"     # pegando o modelo local
        tapas_model = TapasForQuestionAnswering.from_pretrained(path_local, local_files_only=True)
        path_local = "/modelos/google_tokenizer_tapas-large-finetuned-wikisql-supervised"
        tapas_tokenizer = TapasTokenizer.from_pretrained(path_local, local_files_only=True)

        # reader TAPEX
        ## TAPEX
        diretorio = "/modelos/microsoft_tapex-large-finetuned-wtq"
        tapex_tokenizer = TapexTokenizer.from_pretrained(diretorio)
        tapex_model = BartForConditionalGeneration.from_pretrained(diretorio)
        #print(tapex_model)

        ## reranker
        #diretorio = "/modelos/reranker/cross-encoder-stsb-roberta-base"
        #rerank_cross_encoder_model = CrossEncoder(diretorio)


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



## IMPORTANTE:
#primeiro avaliar para posterior comparacao
#para avaliar as respostas:
#se for lista de números:
#    para cada número, remover ',' e transformar em uma unica string,
#se for lista de strings 
#    transformar em uma unica strings
#remover $
#remover %
#remover ()



