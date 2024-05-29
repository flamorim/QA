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
#import llmtable
#import llmqareader
#import evaluate
#import llmquestions

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

def ______preprocess_tables(tables: list):
    processed = []
    # loop through all tables
    for table in tables:
        # convert the table to csv and
        processed_table = "\n".join([table.to_csv(index=False)])
        # add the processed table to processed list
        processed.append(processed_table)
    return processed


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


def rrrrrrre_ranking(question,tables):
    # Now, do the re-ranking with the cross-encoder
    start_time = time.time()
    sentence_pairs = [[question, table] for table in tables]  # montou os i pares pergunta:hit[i]
    cross_encoder_scores = rerank_cross_encoder_model.predict(sentence_pairs) #, show_progress_bar=True)
    
    return(cross_encoder_scores)

def ggggget_top100_tables(list_top100_uid):
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
    
def get_tapas_result(query,results,aggregation_predictions_string):
  queries = [query]
  result = []
  for query, answer, predicted_agg in zip(queries, results, aggregation_predictions_string):
      #print(query)
      if predicted_agg == "NONE":
          result.append(answer)
      else:
          result.append(predicted_agg + " > " + answer)

  return result


def build_reader(retriever_file,device):
            dataset = pd.read_csv(retriever_file, sep=',') #, converters={'top100_table_uid': converter_lista})

            df = pd.DataFrame()

            count_EM_tapex = 0
            count_EM_tapas = 0


            for index, data in dataset.iterrows():
                inp_question   = data['question_txt']
                # pegando a saída do retriever, isto é o primeiro da top100 dele
                answer_table_uid = eval(data['top100_table_uid'])[0].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.').replace('*', '')  

                table = "/data/ott-qa/new_csv/" + answer_table_uid + ".csv"
                table = pd.read_csv(table, sep=',', index_col=None)
                table = table.astype(str)
                real_answer = data['answer_text_gt']

                print('question: ', inp_question)
                print('Real answer: ', real_answer)
                print('usou a tabela gt:', data.top1_flag)
                print("tapex:")
                tapex_answer_text, tapex_generated_tokens = question_answering(inp_question, table, model='tapex')      #***
                print('Resposta: ', tapex_answer_text)
                if real_answer.lower().strip() ==  tapex_answer_text.lower().strip():
                    tapex_top1_flag = True                                                             #***
                    count_EM_tapex +=1
                    print("acertou!")
                else:
                    tapex_top1_flag = False
                    print("errou!")
                print("")

                print("tapas:")
                results, aggregation_predictions_string = question_answering(inp_question, table, model='tapas')
                result = get_tapas_result(inp_question,results,aggregation_predictions_string)
                tapas_answer_text =result[0].strip()
                print('Resposta: ', tapas_answer_text)
                if real_answer.lower().strip() ==  tapas_answer_text.lower().strip():
                    tapas_top1_flag = True
                    count_EM_tapas +=1
                    print("acertou!")
                else:
                    tapas_top1_flag = False  ## precisa tratar
                    print("errou!")
                print("")

                new_data = {'tapex_answer_text'   : tapex_answer_text,
                            'tapex_answer_tokens' : tapex_generated_tokens,
                            'tapex_top1_flag'     : tapex_top1_flag,
                            'tapas_answer_text'   : tapas_answer_text,
                            'tapas_top1_flag'     : tapas_top1_flag,
                            'tapas_agg_string'    : aggregation_predictions_string}

                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                print(index)

                if index % 500 == 0:
                    
                    reader_file = retriever_file.replace('/retriever/','/reader_tapex_tapas/improved/')
                    reader_file = reader_file.replace('csv',f'{index}.csv')
                    

                    df_reader = pd.concat([dataset.reset_index(drop=True), df.reset_index(drop=True)], axis=1)  ## ok
                    df_reader.to_csv(reader_file, sep=',', index=False)       
                    print(f'criado {reader_file}')
                time.sleep(3)
            df_reader = pd.concat([dataset.reset_index(drop=True), df.reset_index(drop=True)], axis=1)  ## ok
            return(df_reader)



#########################################################################

def question_answering(query, table, model):
  if model == "tapex":  # 1024 tokens
    encoding = tapex_tokenizer(table=table, query=query, return_tensors="pt", truncation=True, padding=True, max_length=tapex_model.config.max_position_embeddings, verbose=True)
    outputs = tapex_model.generate(**encoding)
    results = tapex_tokenizer.batch_decode(outputs, skip_special_tokens=True)#.strip()
    generated_tokens = [string.split() for string in results]
    return results[0].strip(), generated_tokens
  elif model == "tapas":
    queries = [query]
    inputs = tapas_tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
    outputs = tapas_model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tapas_tokenizer.convert_logits_to_predictions(
                                                              inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())

    # let's print out the results:
    id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

    results = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            # only a single cell:
            results.append(table.iat[coordinates[0]])   # coordinates indica a coordenada da célula
        else:
            # multiple cells
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
            results.append(", ".join(cell_values))
    return results, aggregation_predictions_string

  return False

#########################################################################


def bbbbuild_reranking(retriever_file, device, run_improve_question):
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
                elif data['table_uid_gt'] in rr['uid'][1:11]:
                    rr_top10 = True
                elif data['table_uid_gt'] in rr['uid'][11:51]:
                    rr_top50 = True
                elif data['table_uid_gt'] in rr['uid'][51:]:
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

                if index % 50 == 0:
                    
                    rr_file = retriever_file.replace('/retriever/','/reranking/')
                    rr_file = rr_file.replace('csv',f'{index}.csv')
                    if run_improve_question == True:
                        rr_file = rr_file.replace('csv',f'{index}.csv')
                        rr_file = rr_file.replace('improved','improved003')

                    df_reranking = pd.concat([dataset.reset_index(drop=True), df.reset_index(drop=True)], axis=1)  ## ok
                    df_reranking.to_csv(rr_file, sep=',', index=False)       
                    print(f'criado {rr_file}')

            df_reranking = pd.concat([dataset.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
            return(df_reranking)



def build_llm_table_opt(retriever_file, device):

    num_candidates = 5  #0
            # fazer aqui a leitura de todas as perguntas e a resposta

    dataset = pd.read_csv(retriever_file, sep=',') #, converters={'top100_table_uid': converter_lista})

    df = pd.DataFrame()


    for index, data in dataset.iterrows():
            inp_question   = data['question_txt']
                # pegando a saída do retriever, isto é o primeiro da top100 dele
            new_data = {}
            new_data['question'] = inp_question
            new_data['model'] = retriever_file
            
            df_per_question = pd.DataFrame()
            for idx in range(num_candidates):  # fazendo para cada uma das topn
                    error = False
                    pos = 'top' + str(idx+1) # de top1 ate top10
                    new_data['position'] = pos

                    answer_table_uid = eval(data['top100_table_uid'])[idx].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.').replace('*', '')  
                    table_uid = "/data/ott-qa/new_csv/" + answer_table_uid + ".csv"
                    new_data['table_uid'] = answer_table_uid

                    prompt_template = build_template_new(num_candidates)
                    prompt =  ChatPromptTemplate.from_template(prompt_template)
                    chain = LLMChain(llm=llm, prompt=prompt,verbose=False)
                    status = 'OK'
                    try:    # problema no LLM, exemplo violar politica PB 
                                    input_data_dict = build_input(inp_question, table_uid)
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
                    tempo_aleatorio = random.randint(7, 12)
                    print(index)
                    print(f'status: {status}')
                    print(new_data['column_names'])
                    print(new_data['column_scores'])

                    print(f"Aguardando por {tempo_aleatorio} segundos...")
                    time.sleep(tempo_aleatorio)
                    df_per_question = pd.concat([df_per_question, pd.DataFrame([new_data])], ignore_index=True)

            df = pd.concat([df, df_per_question], ignore_index=True)
            if index % 360 == 0:
                llm_opt_file = retriever_file.replace('/retriever/','/llm_table_opt/')
                df.to_csv(llm_opt_file, sep=',', index=False)       
                print(f'criado {llm_opt_file}')            
    return(df)


def build_template_new(num_tables):

    question = "{inp_question}"
    table = "{table}"
    prompt_template = f"""
            For the provided question and the provided table, you have to identify the column names in the table that are not relevant \
                for extracting the answer in a question and answer system. \
            For this, you must calculate for each column the relevance <<<score>>>, in a scale from 0 to 1, indicating how relevant the column is. \
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



def build_llm_table_opt_error(retriever_file, device):

    num_candidates = 5  #0
            # fazer aqui a leitura de todas as perguntas e a resposta

    dataset = pd.read_csv(retriever_file, sep=',') #, converters={'top100_table_uid': converter_lista})

    df = pd.DataFrame()


    for index, data in dataset.iterrows():
            inp_question   = data['question_txt']
                # pegando a saída do retriever, isto é o primeiro da top100 dele
            new_data = {}
            new_data['question'] = inp_question
            new_data['model'] = retriever_file
            
            df_per_question = pd.DataFrame()
            for idx in range(num_candidates):  # fazendo para cada uma das topn
                    error = False
                    pos = 'top' + str(idx+1) # de top1 ate top10
                    new_data['position'] = pos

                    answer_table_uid = eval(data['top100_table_uid'])[idx].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.').replace('*', '')  
                    table_uid = "/data/ott-qa/new_csv/" + answer_table_uid + ".csv"
                    new_data['table_uid'] = answer_table_uid

                    prompt_template = build_template_new(num_candidates)
                    prompt =  ChatPromptTemplate.from_template(prompt_template)
                    chain = LLMChain(llm=llm, prompt=prompt,verbose=False)
                    status = 'OK'
                    try:    # problema no LLM, exemplo violar politica PB 
                                    input_data_dict = build_input(inp_question, table_uid)
                                    resposta = chain.invoke(input_data_dict)
                    except:
                                    error = True
                                    status = 'Erro no llm'

                    try:    # problema na saída do LLM, exemplo trocar as chaves do json de saída
                                    output_dict = json.loads(resposta['text'])
                                    new_data['column_names']  = output_dict['column_name']
                                    new_data['column_scores'] = output_dict['score']
                    except:
                                    if error == False:
                                        error = True
                                        status = 'Erro na geração'

                    if error == True:
                            new_data['column_names'] = ['error']
                            new_data['column_scores'] = [0]

                    new_data['status'] = status
                    tempo_aleatorio = random.randint(7, 12)
                    print(index)
                    print(f'status: {status}')
                    print(new_data['column_names'])
                    print(new_data['column_scores'])

                    print(f"Aguardando por {tempo_aleatorio} segundos...")
                    time.sleep(tempo_aleatorio)
                    df_per_question = pd.concat([df_per_question, pd.DataFrame([new_data])], ignore_index=True)

            df = pd.concat([df, df_per_question], ignore_index=True)
            if index % 360 == 0:
                llm_opt_file = retriever_file.replace('/retriever/','/llm_table_opt/')
                df.to_csv(llm_opt_file, sep=',', index=False)       
                print(f'criado {llm_opt_file}')            
    return(df)



def main():

    # path_to_files
    # retriever_source':      '/data/ott-qa/embeddings/',                    #  base line path
    # 'retriever_source':      '/data/ott-qa/embeddings/improved/',             #  nome das colunas sem caracter especial
    #        #'retriever_destination': '/data/ott-qa/retriever/',                     #  base line path
        #'retriever_destination': '/data/ott-qa/retriever/improved001/',          #  nome das colunas sem caracter especial
        #'retriever_destination': '/data/ott-qa/retriever/improved002/',         #  aplicado llm nas questions base line path
     #   'retriever_destination': '/data/ott-qa/retriever/improved003/',         #  nome das colunas sem caracter especial e aplicado llm nas questions base line path
        #'retriever_destination': '/data/ott-qa/embeddings/retriever/improved001', #   perguntas transformadas em sentencas pelo LLM - REFAZER
        #'reranker_roberta_source' : '/data/ott-qa/retriever/',                     # retriever baseline
        #'reranker_roberta_source' : '/data/ott-qa/retriever/improved002/',
        #'reranker_roberta_destination' : '/data/ott-qa/reranking/',
       

    # parametros de configuracao
    with open('/QA/Bert/code/path_to_files.json', 'r') as file:
        path_to_files = json.load(file)

    print(path_to_files)



    run_retriever             = False
    run_improve_question      = False   # llm nas perguntas
    run_reranker_roberta      = True
    run_reranker_marcomini    = False
    run_reranker_marcomini_intro    = False
    run_reader_tapas_tapex    = False
    run_llm_table_opt         = False
    run_llm_rag_and_wikipedia = False
    run_evaluate_answers      = False

    if run_retriever == True:    ## fazendo com os novos embeddings - nome das colunas melhorados
        embeddings = [#'mpnet_table_header_embeddings_cpu_384_512.pkl',
                    #'mpnet_table_header_embeddings_cpu_512_512.pkl']#,
                    #'mpnet_table_section_title_embeddings_cpu_512_512.pkl',
                    #'mpnet_table_section_title_embeddings_cpu_384_512.pkl',
                    'mpnet_table_intro_embeddings_cpu_512_512.pkl',
                    'mpnet_table_embeddings_cpu_512_512.pkl']
        for embedding_file in embeddings:
            embedding_file = path_to_files['retriever_source'] + embedding_file
            #embedding_file =  f'/data/ott-qa/embeddings/improved/{embedding_file}'
            device = 'CPU'
            df_retriever = retriever.build_retriever(embedding_file, "cpu", run_improve_question, llm=llm)

            retriever_file = path_to_files['retriever_destination'] + embedding_file.split('/')[-1]
            retriever_file = retriever_file.replace('.pkl','.csv')
            #if run_improve_question == True:
            #retriever_file = retriever_file.replace('/improved/','/improved003/')

            df_retriever.to_csv(retriever_file,sep=',',index=False)



# Roberta
    if run_reranker_roberta == True:
        diretorio = "/modelos/reranker/cross-encoder-stsb-roberta-base"
        rerank_cross_encoder_model = CrossEncoder(diretorio)
        retriever_output = [#'mpnet_table_section_title_embeddings_cpu_512_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_384_512.csv',
                            #'mpnet_table_intro_embeddings_cpu_512_512.csv',
                            'mpnet_table_embeddings_cpu_512_512.csv']
        for retriever_file in retriever_output:
#            retriever_file =  f'/data/ott-qa/retriever/improved003/{retriever_file}'
            retriever_file = path_to_files['reranker_roberta_source'] + retriever_file
            device = 'CPU'
            df_reranking = rerankingroberta.build_reranking(retriever_file, "cpu", run_improve_question)
            reranking_file = path_to_files['reranker_roberta_destination'] + retriever_file.split('/')[-1]
            df_reranking.to_csv(reranking_file,sep=',',index=False)               

#################################################################

    if run_reranker_marcomini == True:
        diretorio = "/modelos/reranker/cross-encoder-ms-marco-MiniLM-L-6-v2"
        cross_encoder = CrossEncoder(diretorio)
        retriever_output = [#'mpnet_table_section_title_embeddings_cpu_512_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_384_512.csv',
                            'mpnet_table_intro_embeddings_cpu_512_512.csv',
                            'mpnet_table_embeddings_cpu_512_512.csv']
        for retriever_file in retriever_output:
#            retriever_file =  f'/data/ott-qa/retriever/improved003/{retriever_file}'
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
                            'mpnet_table_intro_embeddings_cpu_512_512.csv',
                            'mpnet_table_embeddings_cpu_512_512.csv']
        for retriever_file in retriever_output:
#            retriever_file =  f'/data/ott-qa/retriever/improved003/{retriever_file}'
            retriever_file = path_to_files['reranker_marcominiintro_source'] + retriever_file
            device = 'CPU'
            df_reranking = rerankingmarcoMiniLMintro.build_reranking(retriever_file, "cpu", run_improve_question)
            reranking_file = path_to_files['reranker_marcominiintro_destination'] + retriever_file.split('/')[-1]
            df_reranking.to_csv(reranking_file,sep=',',index=False)

#################################################################

    if run_reader_tapas_tapex == True:
        retriever_output = ['mpnet_table_intro_embeddings_cpu_512_512.csv',
                            'mpnet_table_embeddings_cpu_512_512.csv']#,
                            #'mpnet_table_intro_embeddings_cpu_384_512.csv',
                            #'mpnet_table_embeddings_cpu_384_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_512_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_384_512.csv']
        for retriever_file in retriever_output:
            retriever_file =  f'/data/ott-qa/retriever/improved/{retriever_file}'
            device = 'CPU'
            df_reader = build_reader(retriever_file,"cpu")
            reader_file = retriever_file.replace('/retriever/','/reader_tapex_tapas/improved/')
            df_reader.to_csv(reader_file,sep=',',index=False)       
            print(f'criado {reader_file}')
            print("")


    # otimizacao das tabelas, fazendo um score para cada coluna de relevancia perante
    # a question. Arquivo outpu exclusivo
    if run_llm_table_opt == True:
        retriever_output = ['mpnet_table_intro_embeddings_cpu_512_512.csv',
                            'mpnet_table_embeddings_cpu_512_512.csv',
                            'mpnet_table_intro_embeddings_cpu_384_512.csv',
                            'mpnet_table_embeddings_cpu_384_512.csv',
                            'mpnet_table_section_title_embeddings_cpu_512_512.csv',
                            'mpnet_table_section_title_embeddings_cpu_384_512.csv']

        for retriever_file in retriever_output:
            retriever_file =  f'/data/ott-qa/retriever/{retriever_file}'
            device = 'CPU'
            df_llm_opt = llmtable.build_llm_table_opt(retriever_file,"cpu",llm=llm)
            llm_opt_file = retriever_file.replace('/retriever/','/llm_table_opt/')
            df_llm_opt.to_csv(llm_opt_file, sep=',', index=False)       
            print(f'criado {llm_opt_file}')
            print("")


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







