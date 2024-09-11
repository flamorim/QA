## executa o llm para melhorar as perguntas e também tem as rotinas para ajustar suas colunas

import json
import time, random
import pandas as pd
import utils
import ast

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain




def invoke_evaluate_rewriting(input_data_dict, chain):
            output_dict = {}
        #try:
            resposta = chain.invoke(input_data_dict)
            output_dict = json.loads(resposta['text'])
            output_dict['status_evaluate'] = 'success'
            return output_dict
        #except:
            output_dict['status_evaluate'] = 'failure'
            return output_dict


def build_template_evaluate_rewriting():

    sentence_one = "{sentence_one}"
    sentence_two = "{sentence_two}"
    context      = "{context}"
    

    prompt_template = f"""


    ## ROLE
    You are an knowledge worker very good in writing and understanding text.

    ## TASK
    Your task is to verify semantic textual similarity between two sentences based on the provided context.
    For this, you must calculate a score, in a scale from 0 to 1.
    You will do it in multiple steps:
        1. Read carefully the context
        2. Read the two sentences
        3. First verify if the two senteces are semantic similar
            3.1 If afirmative, verify if the two sentences are semantic similar in the given context
                3.1.1 If afirmative, the score is high
                3.1.2 If negative, the score is medium
            3.2 If negative, the score is low
        5. If the two senteces are not semantic similar, the score is low
    Don't make things up. 
    Let's think step by step.

    ## CONSTRAINTS
    Do not consider any other information besides the sentences and context given

    ## EXAMPLE
    sentence_one: 'Who created the series in which the character of Robert, played by actor Nonso Anozie, appeared?'
    sentence_two: 'The series in which the character of Robert, played by actor Nonso Anozie, appeared was created by'
    context: "Nonso Anozie (born 17 November 1978) is a British actor who has worked on stage, film, and television. He is best known for his role as Tank in RocknRolla, Sergeant Dap in Ender's Game, Abraham Kenyatta in Zoo, Captain of the Guards in Cinderella and Xaro Xhoan Daxos in the HBO television series Game of Thrones."
    score: high value

   ## EXAMPLE
    sentence_one: "what did the 2nd championship win at the Sevens Grand Prix Series for the team with the most top 4 finishes qualify them for ?"
    sentence_two: "The 2nd championship win at the Sevens Grand Prix Series for the team with the most top 4 finishes qualified them for"
    context: "Abhas Kumar Ganguly (born 4 August 1929 - 13 October 1987), better known by his stage name Kishore Kumar (pronunciation (help·info)) was an Indian playback singer, actor, music director, lyricist, writer, director, producer and screenwriter. He is considered as one of the most popular singers of Indian film industry and from soft numbers to peppy tracks to romantic moods, Kumar sang in different genres but some of his rare compositions which were considered classics were lost in time. According to Ashok Kumar, Kumar's success lies in the fact that his voice used to hit the microphone straight at its most sensitive point. Apart from Hindi, he sang in many Indian languages including Bengali, Marathi, Assamese, Gujarati, Kannada, Bhojpuri, Malayalam and Urdu. He has also sung in private albums in several languages especially in Bengali. He won 8 Filmfare Awards for Best Male Playback Singer and holds the record for winning the most Filmfare Awards in that category. He was awarded the Lata Mangeshkar Award by the Madhya Pradesh government in the year 1985-86. In the year 1997, the Madhya Pradesh Government initiated an award called the Kishore Kumar Award as a contribution to Hindi cinema. Recently, Kishore Kumar's unreleased last song was sold for Rs 1560,000 (1.56 million) at the Osian's Cinefan Auction, New Delhi in 2012."
    score: low value

   ## EXAMPLE
    sentence_one: What is the full birth name of the Bradford A.F.C player that only played for the team in 2011 ?
    sentence_two: How many academic staff are at the university in Budapest that has the official abbreviation BME ?
    context: "Bradford City Association Football Club is a professional football club in Bradford, West Yorkshire, England. The team compete in League Two, the fourth tier of the English football league system. They are the only professional football club in England to wear claret and amber, and have worn these colours throughout their history. They have though been known by various nicknames, with the Bantams being the most commonly used nickname as it appears on the current club crest. Supporters hold West Yorkshire derby rivalries with Huddersfield Town and Leeds United, as well as an historic Bradford derby rivalry with the now non-league side Bradford (Park Avenue). The club's home ground is the 25,136-capacity Valley Parade, which was the site of the Bradford City stadium fire on 11 May 1985, which took the lives of 56 supporters. The club was founded in 1903 and immediately elected into the Football League Second Division. Promotion to the top tier followed as they won the 1907-08 Second Division title and then they went on to win the 1911 FA Cup Final, which remains the club's only major honour. They were relegated in 1922 and again in 1927, before winning the Third Division North title in 1928-29. Another relegation in 1937 did allow the club to go on to win the Third Division North Cup in 1939, however a further relegation followed in 1962 to leave the club in the newly created Fourth Division. They secured promotions back into the third tier in 1969 and 1977, but were relegated in 1972 and 1978. They found success in the 1980s under the stewardship of first Roy McFarland and then Trevor Cherry, winning promotion in 1981-82 and following this up with the Third Division title in 1984-85, though they were relegated out of the Second Division in 1990."
    score: low value

   ## INPUTS
    Sentence one: {sentence_one} \
    Sentence two: {sentence_two} \
    Context: {context} \

    ## output
    Format the output as JSON with the following key: \
    score: scale from 0 to 1 \
    """

    return(prompt_template)


def build_evaluate_rewriting(rewriting_file, device, llm):

    df_questions = pd.read_csv(rewriting_file,sep=',')

    ###df_questions = df_questions.drop(df_questions.index[25:])



    question_txt_list = df_questions.question_txt.values.tolist()
    question_opt_list = df_questions.questions_opt.values.tolist() 
    table_intro_list  = df_questions.table_intro.values.tolist()

    prompt_template = build_template_evaluate_rewriting()                                  ##
    prompt_generator =  ChatPromptTemplate.from_template(prompt_template)
    chain_question_opt = LLMChain(llm=llm, prompt=prompt_generator,verbose=False)

    new_evaluate_status_list = []
    new_score_list           = []

    index = 0

    for sentence_one, sentence_two, context in zip(question_txt_list, question_opt_list, table_intro_list):
        
        input_data_dict = {}
        input_data_dict['sentence_one']   = sentence_one
        input_data_dict['sentence_two']   = sentence_two
        input_data_dict['context']        = context
        
        output_dict = invoke_evaluate_rewriting(input_data_dict, chain_question_opt)

        if output_dict['status_evaluate'] == 'failure':
            new_score_list.append(float('-1.0'))
            new_evaluate_status_list.append('failure')
        else:
            new_score_list.append(float(output_dict['score']))
            new_evaluate_status_list.append('success')


        print(output_dict['score'])
        
        tempo_aleatorio = random.randint(1, 3)
        print(f"Aguardando por {tempo_aleatorio} segundos...")
        print()
        time.sleep(tempo_aleatorio)

        if index % 250 == 0:

                out_file = rewriting_file.replace('csv',f'{index}.teste.csv')
                df_out = pd.DataFrame()
                df_out = df_questions.copy()
                df_out = df_out.drop(df_out.index[index+1:])  ## teste 
                df_out['score'] = new_score_list
                df_out['status_evaluate'] = new_evaluate_status_list

                #llm_opt_file = retriever_file.replace('/retriever/','/llm_table_opt/')
                df_out.to_csv(out_file, sep=',', index=False)       
                print(f'criado {out_file}')

        index += 1

    df_questions['score'] = new_score_list
    df_questions['status_evaluate'] = new_evaluate_status_list

    return df_questions


def invoke_evaluate_rewriting(input_data_dict, chain):
            output_dict = {}
        #try:
            resposta = chain.invoke(input_data_dict)
            output_dict = json.loads(resposta['text'])
            output_dict['status_evaluate'] = 'success'
            return output_dict
        #except:
            output_dict['status_evaluate'] = 'failure'
            return output_dict


def build_template_evaluate_rewriting():

    sentence_one = "{sentence_one}"
    sentence_two = "{sentence_two}"
    context      = "{context}"
    

    prompt_template = f"""


    ## ROLE
    You are a knowledge worker very good in question and answer systems.

    ## TASK
    Your task is to verify semantic textual similarity between two answers from a question based on the provided context.
    For this, you must calculate a score, in a scale from 0 to 1.
    You will do it in multiple steps:
        1. Read carefully the context
        2. Read the two answers
        3. First verify if the two answers are semantic similar
            3.1 If afirmative, verify if the two answers are semantic similar in the given context
                3.1.1 If afirmative, the score is high
                3.1.2 If negative, the score is medium
            3.2 If negative, the score is low
    Don't make things up. 
    Let's think step by step.

    ## CONSTRAINTS
    Do not consider any other information besides the sentences and context given

    ## EXAMPLE
    sentence_one: 'Nonso Anozie'
    sentence_two: 'Emerson Fitipaldi'
    context: "Nonso Anozie (born 17 November 1978) is a British actor who has worked on stage, film, and television. He is best known for his role as Tank in RocknRolla, Sergeant Dap in Ender's Game, Abraham Kenyatta in Zoo, Captain of the Guards in Cinderella and Xaro Xhoan Daxos in the HBO television series Game of Thrones."
    score: high value

    ## EXAMPLE
    sentence_one: "the second position"
    sentence_two: "swiming"
    context: "Abhas Kumar Ganguly (born 4 August 1929 - 13 October 1987), better known by his stage name Kishore Kumar (pronunciation (help·info)) was an Indian playback singer, actor, music director, lyricist, writer, director, producer and screenwriter. He is considered as one of the most popular singers of Indian film industry and from soft numbers to peppy tracks to romantic moods, Kumar sang in different genres but some of his rare compositions which were considered classics were lost in time. According to Ashok Kumar, Kumar's success lies in the fact that his voice used to hit the microphone straight at its most sensitive point. Apart from Hindi, he sang in many Indian languages including Bengali, Marathi, Assamese, Gujarati, Kannada, Bhojpuri, Malayalam and Urdu. He has also sung in private albums in several languages especially in Bengali. He won 8 Filmfare Awards for Best Male Playback Singer and holds the record for winning the most Filmfare Awards in that category. He was awarded the Lata Mangeshkar Award by the Madhya Pradesh government in the year 1985-86. In the year 1997, the Madhya Pradesh Government initiated an award called the Kishore Kumar Award as a contribution to Hindi cinema. Recently, Kishore Kumar's unreleased last song was sold for Rs 1560,000 (1.56 million) at the Osian's Cinefan Auction, New Delhi in 2012."
    score: low value

    ## INPUTS
    Sentence one: {sentence_one} \
    Sentence two: {sentence_two} \
    Context: {context} \

    ## output
    Format the output as JSON with the following key: \
    score: scale from 0 to 1 \
    """

    return(prompt_template)



def main():

    import os
    import torch

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    max_corpus_size = 0 # 500 #25  # 0 significa sem restricao
    #dataset_path = "data/wikipedia/simplewiki-2020-11-01.jsonl.gz"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    ConversationBuffer = True



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
    #if  ConversationBuffer == False:
    #    memory = ConversationBufferWindowMemory(k=1)
    #else:
    #    memory = ConversationBufferWindowMemory()


    rewriting_file = '/QA/Bert/data/ott-qa/question_rewriting/build_improve_questions.csv'  # só para pegar os textos
    df_questions = pd.read_csv(rewriting_file,sep=',')
    #question_txt_list = df_questions.question_txt.values.tolist()
    table_intro_list  = df_questions.table_intro.values.tolist()
    #question_id_list = df_questions.question_id.values.tolist() 


    reader_file = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512_evaluated.csv'
    df_reader   = pd.read_csv(reader_file,sep=',')
    df_reader['table_intro_gt'] = table_intro_list  

    df_reader = df_reader.drop(df_reader.index[:2200])  ## aqui 

    # 1) question_txt_list = vai ser a answer gt
    question_txt_list = df_reader.answer_text_gt.tolist()  # OK

    # 2) question_opt_list = vai ser a answer do tapex
    question_opt_list = []
    for index, data in df_reader.iterrows():
        answer_text_list = eval(data['tapex_answer_text'])
        question_opt_list.append(answer_text_list[0])

    # 3) table_intro_list  = vai ser a introducao
    table_intro_list  = df_reader.table_intro_gt.values.tolist()

    prompt_template = build_template_evaluate_rewriting()                                  ##
    prompt_generator =  ChatPromptTemplate.from_template(prompt_template)
    chain_question_opt = LLMChain(llm=llm, prompt=prompt_generator,verbose=False)

    new_evaluate_status_list = []
    new_score_list           = []

    index = 0

#    for sentence_one, sentence_two, context in zip(question_txt_list, ast.literal_eval(question_opt_list[0]), table_intro_list):
    for sentence_one, sentence_two, context in zip(question_txt_list, question_opt_list, table_intro_list):
        
        input_data_dict = {}
        input_data_dict['sentence_one']   = sentence_one
        input_data_dict['sentence_two']   = sentence_two
        input_data_dict['context']        = context
        
        output_dict = invoke_evaluate_rewriting(input_data_dict, chain_question_opt)

        if output_dict['status_evaluate'] == 'failure':
            new_score_list.append(float('-1.0'))
            new_evaluate_status_list.append('failure')
        else:
            new_score_list.append(float(output_dict['score']))
            new_evaluate_status_list.append('success')


        print(output_dict['score'])
        print(sentence_one)
        print(sentence_two)
        
        tempo_aleatorio = random.randint(1, 3)
        print(f"Aguardando por {tempo_aleatorio} segundos...")
        print()
        time.sleep(tempo_aleatorio)
        index += 1

    #    if index % 50 == 0:
            #out_file = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512_evaluated_sts.csv'
            #out_file = out_file.replace('.csv',f'.{index}.csv')
            #df_out = pd.DataFrame()
            #df_out = df_questions.copy()
            #df_out = df_out.drop(df_out.index[:2214-index])  ## teste 
            #df_out['sts_score'] = new_score_list
            #df_out['sts_status_evaluate'] = new_evaluate_status_list

                #llm_opt_file = retriever_file.replace('/retriever/','/llm_table_opt/')
            #df_out.to_csv(out_file, sep=',', index=False)       
            #print(f'criado {out_file}')

    
    df_reader['sts_score'] = new_score_list
    df_reader['sts_status_evaluate'] = new_evaluate_status_list
    
    fout = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512_evaluated_sts.csv'
    df_reader.to_csv(fout, sep=',', index=False)

    exit()



######################################################################


debug_mode = True

if __name__ == '__main__':
    if debug_mode == True:
        import debugpy
        debugpy.listen(7011)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    print('hello 2')
    i = 1  # local para breakepoint do debuger
    print('hello 3')


    fin = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512_evaluated_acc_f1_sts.csv'
    dataset = pd.read_csv(fin,sep=',')
    #print(dataset.columns)

    index = dataset.shape[0]
    print('resumo acuracia:')
    em      = (dataset['reader_exact_match_list'].sum()/index)
    top10   = (dataset['reader_top10_list'].sum()/index)
    top50   = (dataset['reader_top50_list'].sum()/index)
    top100  = (dataset['reader_top100_list'].sum()/index)
    print(em)
    print(em + top10)
    print(em + top10 + top50)
    print(em + top10 + top50 + top100)

    print('resumo f1:')
    f1 = (dataset['reader_f1'].sum()/index)
    print(f1)
    print()

    print('resumo f1:')
    sts = (dataset['sts_score'].sum()/index)
    print(sts)
    print()

#resumo acuracia:
#0.027551942186088526
#0.07091237579042457
#0.12601626016260162
#0.16169828364950314
#resumo f1:
#0.0499817832473659

#resumo f1:
#0.2160569105691057



def concatena():
    
    part1 = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512_evaluated_sts.2200.correto.csv'  # só para pegar os textos
    df_1 = pd.read_csv(part1,sep=',')

    part2 = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512_evaluated_sts.14.csv'  # só para pegar os textos
    df_2 = pd.read_csv(part2,sep=',')

    df_2 = df_2.drop(df_2.index[:1])
    df = pd.concat([df_1.reset_index(drop=True), df_2.reset_index(drop=True)], axis=0)  ## ok
    print(df.shape)

    part3 = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512_evaluated.csv'    
    df_3 = pd.read_csv(part3,sep=',')

    df_final = pd.concat([df_3.reset_index(drop=True), df.reset_index(drop=True)], axis=1)  ## ok
    print(df_final.shape)

    fout = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512_evaluated_acc_f1_sts.csv'
    df_final.to_csv(fout, sep=',', index=False)


    #main()
    #concatena()



