## executa o llm para melhorar as perguntas e também tem as rotinas para ajustar suas colunas

import json
import time, random
import pandas as pd
import numpy as np
import utils
import os
import requests
from bs4 import BeautifulSoup

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain




def invoke_question_opt(input_data_dict, chain):
            output_dict = {}
        #try:
            resposta = chain.invoke(input_data_dict)
            output_dict = json.loads(resposta['text'])
            output_dict['status_evaluate'] = 'success'
            return output_dict
        #except:
            output_dict['status_evaluate'] = 'failure'
            return output_dict




def build_template_question_opt():

    question = "{inp_question}"

    prompt_template = f"""


    ## ROLE

    You are a teacher very good in writing text

    ## TASK
    Your task is to generate a grammatically correct affirmative or negative sentence based on the provided question.
    The task is not to generate the answer and do not add any information correlated to the answer.

    ## EXAMPLE
    question: 'Who created the series in which the character of Robert, played by actor Nonso Anozie, appeared?'
    sentence: "The series in which the character of Robert, played by actor Nonso Anozie, appeared was created by"

    Question:

    {question}

    ## OUTPUT

    Format the output as JSON with the 'question' and 'sentence' as keys and theis contents as values:

            "question": <<<question>>> ,
            "sentence": <<<sentence>>>
    """



    return(prompt_template)


def build_template_question_opt_v2():

    question = "{inp_question}"

    prompt_template = f"""


    ## ROLE

    You are a bot that works for question and answer system

    ## TASK
    Your task is to convert a question to a grammatically correct affirmative form sentence.
    The task is not to generate the answer and do not add any information correlated to the answer.

    ## EXAMPLE
    question: 'Who created the series in which the character of Robert, played by actor Nonso Anozie, appeared'
    sentence: "The series in which the character of Robert, played by actor Nonso Anozie, appeared was created by"

    ## Question:

    {question}

    ## OUTPUT

    Format the output as JSON with the 'question' and 'sentence' as keys and theis contents as values:

            "question": <<<question>>> ,
            "sentence": <<<sentence>>>
    """
    return(prompt_template)

def build_template_question_opt_v3():

    question = "{inp_question}"

    prompt_template = f"""


    ## ROLE
    You are a bot that works for question and answer system

    ## TASK
    Your task is to convert a question to an affirmative form sentence.
    Let's think step by step.
    
    ## CONSTRAINTS
    Affirmative form sentences can not end with a question mark.
    
    ## EXAMPLE
    question: 'Who created the series in which the character of Robert, played by actor Nonso Anozie, appeared?'
    sentence: "The series in which the character of Robert, played by actor Nonso Anozie, appeared was created by"

    ## Question:

    {question}

    ## OUTPUT

    Format the output as JSON with the 'question' and 'sentence' as keys and theis contents as values:

            "question": <<<question>>> ,
            "sentence": <<<sentence>>>
    """
    return(prompt_template)


def build_improve_questions(questions, device, llm):
    #return(questions)

    prompt_template = build_template_question_opt_v3()                                  ##
    prompt_generator =  ChatPromptTemplate.from_template(prompt_template)
    chain_question_opt = LLMChain(llm=llm, prompt=prompt_generator,verbose=False)

    new_questions_status = []
    new_questions        = []

    for index, data in enumerate(questions):
        
        #inp_question   = data['question_txt']


        input_data_dict = {}
        input_data_dict['inp_question']   = data
        
        #input_data_dict = utils.build_input_questions(input_data_dict)
        output_dict = invoke_question_opt(input_data_dict, chain_question_opt)

        evaluate_output = {}
        if output_dict['status_evaluate'] == 'failure':
            new_questions.append(data)
            new_questions_status.append('failure')
                       
        else:
            new_questions.append(output_dict['sentence'].replace('<<<','').replace('>>>',''))
            new_questions_status.append('success')

        print(index)
        print(output_dict['question'])
        print(output_dict['sentence'])
        
        tempo_aleatorio = random.randint(1, 4)
        print(f"Aguardando por {tempo_aleatorio} segundos...")
        print()
        time.sleep(tempo_aleatorio)

    return new_questions, new_questions_status


#def evaluate_questions(questions, model):
#    #return(questions)
#    df_questions_improved = pd.read_csv('/data/ott-qa/question_rewriting/build_improve_questions_old.csv',sep=',')       
#            questions_opt = df_questions_improved['questions_opt'].to_list()


def fix_ids():
    with open('/data/ott-qa/released_data/dev.json', 'r') as f:    # perguntas e respostas
        data = json.load(f)
    dataset = utils.convert_dataset(data)
    question_ids = dataset.question_id.values.tolist()

    df_questions_improved = pd.read_csv('/data/ott-qa/question_rewriting/build_improve_questions_old.csv',sep=',')
    df_questions_improved = df_questions_improved.rename(columns={df_questions_improved.columns[0]: 'indice'})
    df_questions_improved['question_id'] = question_ids
    cols = list(df_questions_improved.columns)
    # Mover a última coluna para a segunda posição
    cols.insert(1, cols.pop())
    # Reordenar o DataFrame com as novas posições das colunas
    df_questions_improved = df_questions_improved[cols]



    df_passages = pd.read_csv('/data/ott-qa/retriever/mpnet_table_embeddings_cpu_512_512.csv',sep=',')
    passages = df_passages.question_id.values.tolist()
    #df_questions_improved = df_questions_improved.rename(columns={df_questions_improved.columns[0]: 'indice'})
    df_ved['question_id'] = question_ids
    cols = list(df_questions_improved.columns)
    # Mover a última coluna para a segunda posição
    cols.insert(1, cols.pop())
    # Reordenar o DataFrame com as novas posições das colunas
    df_questions_improved = df_questions_improved[cols]


    #/QA/Bert/data/ott-qa/retriever/improved003/mpnet_table_embeddings_cpu_512_512.csv



    #csv_name = '/data/ott-qa/question_rewriting/build_improve_questions.csv'
    #df_questions_improved.to_csv(csv_name,index=False)


def get_table_id():
    #so para acrescentar uma coluna no arquivo build_improve_questions.csv
    with open('/data/ott-qa/released_data/dev.json', 'r') as f:    # perguntas e respostas
        data = json.load(f)
    dataset = utils.convert_dataset(data)
    
    
    table_uid_list = dataset.table_uid_gt.values.tolist()

    
    csv_name = '/data/ott-qa/question_rewriting/build_improve_questions.csv'  ## 2215 perguntas
    df_questions = pd.read_csv(csv_name,sep=',')
    df_questions['table_uid'] = table_uid_list

    csv_name = '/data/ott-qa/question_rewriting/new_build_improve_questions.csv'
    df_questions.to_csv(csv_name,index=False)


def get_question():
    #so para acrescentar uma coluna no arquivo build_improve_questions.csv

    with open('/data/ott-qa/released_data/dev.json', 'r') as f:    # perguntas e respostas
        data = json.load(f)
    dataset = utils.convert_dataset(data)
    
    
    question_list = dataset.question_txt.values.tolist()

    
    csv_name = '/data/ott-qa/question_rewriting/final_build_improve_questions.csv'  ## 2215 perguntas
    df_questions = pd.read_csv(csv_name,sep=',')
    df_questions['question_txt'] =question_list


    cols = list(df_questions.columns)
    # Mover a última coluna para a segunda posição
    cols.insert(2, cols.pop())
    # Reordenar o DataFrame com as novas posições das colunas
    df_questions = df_questions[cols]


    #csv_name = '/data/ott-qa/question_rewriting/final02_build_improve_questions.csv'
    #df_questions.to_csv(csv_name,index=False)


def get_url_intro(fname):
        #so para acrescentar as colunas url e intro no arquivo build_improve_questions.csv

    PATH_TO_JSON = '/data/ott-qa/traindev_tables.json'   ## 8891 tabelas

    with open(PATH_TO_JSON, 'r') as f:    #traindev_tables.json
        json_data = json.load(f)

    table_intro_list  = []
    table_url_list    = []
    table_uid_list   = []
    table_idx_list    = []
    count = 0

    for key,value in json_data.items():

        intro = value["intro"]
        table_intro_list.append(intro)

        url = value["url"]
        table_url_list.append(url)

        uid = value['uid'].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.').replace('*', '')
        table_uid_list.append(uid)

        table_idx_list.append(count)
        if ((count % 100) == 0):
            print(count)
        count +=1


    csv_name = '/data/ott-qa/question_rewriting/improved_questions.csv'  ## 2215 perguntas
    df_questions = pd.read_csv(fname,sep=',')


    new_table_intro = []
    new_table_url = []

    for index, question in df_questions.iterrows():
        table_uid   = question['table_uid']
        idx1 = table_uid_list.index(table_uid)

        table_into = table_intro_list[idx1]
        new_table_intro.append(table_into)

        table_url = table_url_list[idx1]
        new_table_url.append(table_url)


    df_questions['table_url']   = new_table_url
    df_questions['table_intro'] = new_table_intro

    #df_questions = df_questions.rename(columns={'table_uid': 'table_uid'})

    df_questions.columns
    csv_name = '/data/ott-qa/question_rewriting/NEW_improved_questions.csv'
    df_questions.to_csv(fname,index=False)

def remove_symbols(text):
    return text.replace('<<<', '').replace('>>>', '')

def clean_gpt(fname):
    #remover as ocorrencias de <<< ou >>> nas respostas do gpt
    #fin = '/data/ott-qa/question_rewriting/222improved_questions.csv'
    df = pd.read_csv(fname,sep=',')
    df['questions_opt'] = df['questions_opt'].apply(remove_symbols)

    fout = '/data/tat-qa/question_rewriting/222improved_questions.csv'
    df.to_csv(fname, sep=',', index=False)       
    print(f'refeito: {fname}')

def get_first_paragraph(url):        
        # Faz a solicitação HTTP para a URL
        response = requests.get(url)
        response.raise_for_status()  # Levanta um erro se a solicitação falhar

        # Analisa o conteúdo HTML da página
        soup = BeautifulSoup(response.content, 'html.parser')

        # Encontra o primeiro parágrafo no conteúdo principal da página
        content = soup.find('div', {'class': 'mw-parser-output'})

        #first_paragraph = content.find('p').get_text()
        paragraphs = content.find_all('p')

        # Itera sobre os parágrafos até encontrar um não vazio
        text = ''
        for paragraph in paragraphs:
            text = paragraph.get_text().strip()
            if text:  # Verifica se o parágrafo não está vazio
                print(text)
                break
        return(text)

def is_empty_or_nan(var):
    if var is None:
        return True
    if isinstance(var, str) and not var.strip():
        return True
    if isinstance(var, (list, tuple, set, dict)) and not var:
        return True
    if isinstance(var, (int, float)) and np.isnan(var):
        return True
    return False

def fill_intro(row):
    #if not row['table_intro']:
    if is_empty_or_nan(row['table_intro']):
        #if not row['table_intro']:
            text = get_first_paragraph(row['table_url'])
            if text == '':
                text = np.nan
            return text
    return row['table_intro']
    

def fix_empty_intro(fname):
    df = pd.read_csv(fname,sep=',')
    #df = df.drop(df.index[:270])

    df['table_intro'] = df.apply(fill_intro, axis=1)
    print(df.shape)
    df = df.dropna(subset=['table_intro'])
    print(df.shape)
    fout = '/data/ott-qa-opt/question_rewriting/improved_questions.csv'
    df.to_csv(fout, sep=',', index=False)       
    print(f'refeito: {fout}')


def main():
    print("hello")
    #fix_ids()

    # LLM generator
    from langchain.chat_models import AzureChatOpenAI
    from openai.api_resources.abstract import APIResource
    #from openai.resources.abstract import APIResource
    from langchain.document_loaders import CSVLoader

    os.environ["OPENAI_API_KEY"] = '72b26ee264b5440ca36cdf717ee80712'
    os.environ["OPENAI_API_BASE"] = 'https://api.petrobras.com.br'
    os.environ["OPENAI_API_VERSION"] = '2023-03-15-preview'
    os.environ["OPENAI_API_TYPE"] = 'azure'
    os.environ["REQUESTS_CA_BUNDLE"] = "/nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/petrobras-openai/petrobras-ca-root.pem"
    APIResource.azure_api_prefix = 'ia/openai/v1/openai-azure/openai'
    print(os.environ["REQUESTS_CA_BUNDLE"])

    # LLM Model
    llm = AzureChatOpenAI(
        deployment_name="gpt-35-turbo-16k-petrobras",
        model_name="gpt-3.5-turbo-16k",
        temperature=0,
    )

    with open('/data/tat-qa/released_data/dev.json', 'r') as f:    # perguntas e respostas
        data = json.load(f)
    dataset = utils.convert_dataset(data)

    #dataset = dataset.drop(dataset.index[650:])
    #dataset = dataset.drop(dataset.index[:5900])

    questions = dataset.question_txt.values.tolist()

    df_output = pd.DataFrame()
    questions_opt,questions_opt_status = build_improve_questions(questions, device='CPU', llm=llm)

    df_output['question_id']          = dataset.question_id.values.tolist()
    df_output['question_txt']         = dataset.question_txt.values.tolist()
    df_output['questions_opt']        = questions_opt
    df_output['questions_opt_status'] = questions_opt_status
    df_output['table_uid']            = dataset.table_uid_gt.values.tolist()
    #df_output['table_url']            = dataset.table_uid_gt.values.tolist()
    #df_output['table_intro']          = dataset.table_uid_gt.values.tolist()
    #df_output['score']                = dataset.table_uid_gt.values.tolist()
    #df_output['status_evaluate']      = dataset.table_uid_gt.values.tolist()

    output_file = '/data/tat-qa/question_rewriting/improved_questions.csv'
    df_output.to_csv(output_file, sep=',', index=False)       
    print(f'criado: {output_file}')
    clean_gpt(output_file)
    print(f'removido chars')
    #get_url_intro(output_file)
    #print(f'inserido url e introducao')

def remove_questions(file_source, to_remove):
    df = pd.read_csv(file_source, sep=',')
    df_to_remove = pd.read_csv(to_remove, sep=',')
    list_to_remove = df_to_remove['question_id'].to_list()

    print(df.shape)
    df_filtered = df[~df['question_id'].isin(list_to_remove)]
    print(df_filtered.shape)
    new_file = file_source.replace(".csv","_filtered.csv")
    df_filtered.to_csv(new_file, sep=',', index=False)


if __name__ == '__main__':
    main()
    #fix_empty_intro('/data/ott-qa-opt/question_rewriting/improved_questions.csv')

# REMOVIDAS AS SEGUINTES QUESTIONS DO DATASET POIS FAZEM REFERENCIA A UMA PAGINA SEM TEXTO
#5cf37c4385dae623 nao tem intro
#970293d95c911646 nao tem intro

    fin = '/data/tat-qa/question_rewriting/evaluate_improved_questions.csv'
    to_remove = '/data/tat-qa/remover.2024.07.30.csv'
    remove_questions(fin, to_remove)
    print('fim')
