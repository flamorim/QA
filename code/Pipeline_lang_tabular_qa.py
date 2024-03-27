#from tabfact import get_embeddings
#quais são as atividades que são medidas em pes cubicos?
# qual é a atividade que dá maior produção de barris por dia?
#qual é Capacidade de operação de refino?
#quem é Obama?
#what nationality is mack kalvin?
#who is john dramani mahama?


import torch
import os
import pickle
import time
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
    embedding_file = f'table_embeddings_{device}.pkl'
    embedding_file = path + embedding_file
    # Check if embedding cache path exists
    if not os.path.exists(embedding_file):
        print(f'Embeddings nao encontrado em cache {embedding_file}')
        print('abortando')
        exit()
    print(f'Embeddings cache {embedding_file} encontrado localmente...')
    print("Aguarde carregando....")

    with open(embedding_file, "rb") as fIn:    ## erro gpu
            cache_data = pickle.load(fIn)
            tables = cache_data["tables"]
            tables_embedding  = cache_data["embeddings"]
            tables_sentence   = cache_data["tables_sentence"] 
            tables_title    = cache_data["tables_title"]
            tables_file_name   = cache_data["tables_url"]
    if max_corpus_size != 0:
            tables             = tables[0:max_corpus_size]
            tables_embedding   = tables_embedding[0:max_corpus_size]
            tables_sentence    = tables_sentence[0:max_corpus_size]
            tables_title       = tables_title[0:max_corpus_size]
            tables_file_name   = tables_file_name[0:max_corpus_size]
    print("")
    print("Corpus loaded with {} tables / embeddings".format(len(tables)))
    print(f'dimensao do vetor : {len(tables_embedding[0])}')
    print(f'Table embeddings load took {(time.time() - start_time)} seconds')

    return tables_embedding, tables_sentence, tables, tables_file_name


def get_embeddings_sentences(model_name, path, max_corpus_size):
    # Some local file to cache computed embeddings

    start_time = time.time()
    embedding_file = f'table_text_embeddings_{device}.pkl'
    embedding_file = path + embedding_file
    # Check if embedding cache path exists
    if not os.path.exists(embedding_file):
        print(f'Embeddings nao encontrado em cache {embedding_file}')
        print('abortando')
        exit()
    print(f'Embeddings cache {embedding_file} encontrado localmente...')
    print("Aguarde carregando....")

    with open(embedding_file, "rb") as fIn:    ## erro gpu
            cache_data = pickle.load(fIn)
            tables = cache_data["tables"]
            tables_embedding  = cache_data["embeddings"]
            tables_sentence   = cache_data["tables_sentence"] 
            tables_title    = cache_data["tables_title"]
            tables_file_name   = cache_data["tables_url"]
    if max_corpus_size != 0:
            tables             = tables[0:max_corpus_size]
            tables_embedding   = tables_embedding[0:max_corpus_size]
            tables_sentence    = tables_sentence[0:max_corpus_size]
            tables_title       = tables_title[0:max_corpus_size]
            tables_file_name   = tables_file_name[0:max_corpus_size]
    print("")
    print("Corpus loaded with {} tables / embeddings".format(len(tables)))
    print(f'dimensao do vetor : {len(tables_embedding[0])}')
    print(f'Table + sentences embeddings load took {(time.time() - start_time)} seconds')

    return tables_embedding, tables_sentence, tables, tables_file_name


def cosine_similarity(inp_question,corpus_embeddings,num_candidates):

    start_time = time.time()
    question_embedding = retriever_biencoder_model.encode(inp_question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=num_candidates)
    hits = hits[0]  # Get the hits for the first query
    print("")
    print('---------------------------------------------------------------------------')
    print(f'Cosine-Similarity para top-{num_candidates} levou {(time.time() - start_time)} seconds')
    #print(f'imprimindo Top 5 dos {num_candidates} hits with cosine-similarity:')
    #for hit in hits[0:5]:
    #    print("\t{:.3f}\t{}".format(hit["score"], corpus_tables[hit["corpus_id"]]))

    return hits


def build_templateNAO(num_tables):

    docs = ''
    for count in range(num_tables):
        doc = f"doc{count+1}"
        doc = "{" + doc + "},"
        docs = docs + doc

    docs = docs[0:-1]
    question = "{inp_question}"
    prompt_template = f"""You are a very smart oil and gas engineer working at Petrobras in Brazil.\
            Your task is generate a technical and comprehensive answer to the question "{question}" \
            based on the tables {docs} you will receive. \
            Each table has a score showing how relevant the table is about the question and it is {scores}. \
            You do not need to use all tables, but the ones that realy aggregate information and \ 
            in case of any doubt, always chose tables with highest scores.
            
            First you have to translate the "{question}"qu to english and then think about the answer. \

            When you don't know the answer to the question you admit\
            that you don't know and your answer must be "Sory, I do not have this information." \
            
            When you know the answer, it must write it in <<<answer>>> and in the same language of the question, \
            not the language of the documents. \
            Each table that provided valued information, you must write its table name in <<<table_name>>>, file name in <<<file_name>>> and its score number at <<<score>>> \
            Each table where the {question} is not mentioned, don´t include it in the answer. \
            The answer must have two parts. \
            The firt part has to be like: \
                Relevant tables: <<<table_name>>>, file <<<file_name>>>\

            the second part has to be like: \
                    -------------------- \
                    Question: {question} \
                    Answer: <<<answer>>> \
            Do not add any other information besides the {question}, <<<table_name>>>, <<<answer>>>.  \

                    """

   
    return(prompt_template)

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




def build_input(inp_question, hits, tables_file_name, corpus_tables):
    input_for_chain = {}
    input_for_chain['inp_question'] = inp_question

    for count,hit in enumerate(hits):
        tb = corpus_tables[hit["corpus_id"]]
        tb.to_csv('/data/ott-qa/csv/temp/tab1.csv',index=False,sep='#')
        loader = CSVLoader(file_path='/data/ott-qa/csv/temp/tab1.csv',
                        csv_args={"delimiter": "#"})
        doc = loader.load()
        doc_n = f"doc{count+1}"
        input_for_chain[doc_n] = doc

    return(input_for_chain)
    ################# ATENCAO ######################
    for count,hit in enumerate(hits):
        url = tables_file_name[hit["corpus_id"]].split('wiki/')[1]
        url = url.replace('/','_')
        url = url + '.csv'
        file = '/data/ott-qa/csv/' + url

        #file = '/data/ott-qa/csv/' + tables_file_name[hit["corpus_id"]]
        loader = CSVLoader(file_path=file,
                        csv_args={"delimiter": "#"})
        doc = loader.load()
        doc_n = f"doc{count+1}"
        input_for_chain[doc_n] = doc
   
        #score_n = f"score{count+1}"                     RETIRADO O SCORE
        #input_for_chain[score_n] = hit["score"]

    #print(input)
    return(input_for_chain)

#{'inp_question':inp_question,'doc1':doc1,'doc2':doc2,'doc3':doc3}



def main():

    tables_embedding, tables_sentence, corpus_tables, tables_file_name = get_embeddings(retriever_model_name, embedding_path, max_corpus_size)
    sentences_tables_embedding, tables_sentence, corpus_tables, tables_file_name = get_embeddings_sentences(retriever_model_name, embedding_path, max_corpus_size)


    while True:


    # 0 - Question
        inp_question = input("Entre com a question: ")
        num_candidates = int(input("Entre com numero para Top-K candidatos: "))

        # 1 - calcula similaridade - so tabelas
        hits = cosine_similarity(inp_question, tables_embedding, num_candidates)

        print("Embedding so com tabelas")
        print(f'lista dos top-{num_candidates} documentos')
        for hit in hits:
            print(f'id {hit["corpus_id"]} | {tables_file_name[hit["corpus_id"]]} | score {hit["score"]}')
        print('---------------------------------------------------------------------------')



        # 1 - calcula similaridade - tabelas e sentenças
        hits_sentences = cosine_similarity(inp_question, sentences_tables_embedding, num_candidates)

        print("Embedding com tabelas e sentenças")
        print(f'lista dos top-{num_candidates} documentos')
        for hit in hits_sentences:
            print(f'id {hit["corpus_id"]} | {tables_file_name[hit["corpus_id"]]} | score {hit["score"]}')
        print('---------------------------------------------------------------------------')

        
        ##### prompt / LLM
        ###########################################################################
        #file1 = '/data/DADOSDEPRODUCAO-consolidado.csv'
        #loader = CSVLoader(file_path=file1,
        #                    csv_args={"delimiter": "#"})
        #doc1 = loader.load()#

        #file2 = '/data/METRICASDESUSTENTABILIDADE.csv'
        #loader = CSVLoader(file_path=file2,
        #                    csv_args={"delimiter": "#"})
        #doc2 = loader.load()##

        #file3 = '/data/PROJETOSDEDESINVESTIMENTO.csv'
        #loader = CSVLoader(file_path=file3,
        #                    csv_args={"delimiter": "#"})
        #doc3 = loader.load()


        prompt_template = build_template_new(num_candidates)
        prompt =  ChatPromptTemplate.from_template(prompt_template)
            
        #chain = LLMChain(llm=llm, prompt=prompt,verbose=False, memory=memory)
        chain = LLMChain(llm=llm, prompt=prompt,verbose=True)

        input_data_dict = build_input(inp_question, hits, tables_file_name, corpus_tables)
        start_time = time.time()
        resposta = chain.invoke(input_data_dict)

        question_schema = ResponseSchema(name="question",
                                description = "Question on native language.")
        question_english_schema = ResponseSchema(name="question_english",
                                    description = "Question translated to english")
        #relevant_tables_schema = ResponseSchema(name="relevant_tables",
        #                        description = "Tables relevat to get the answer")

        file_names_schema = ResponseSchema(name="source",
                                description = "Document file name relevant to get the answer")

        answer_schema = ResponseSchema(name="answer",
                                description = "Answer for the question")
        response_schemas = [question_schema,
                        question_english_schema,
        #                relevant_tables_schema,
                        file_names_schema,
                        answer_schema]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        print(f'LLM para top-{num_candidates} levou {(time.time() - start_time)} seconds')
        output_dict = output_parser.parse(resposta['text'])

        for k, v in output_dict.items():
            print(k,v)

        print('----------------READER TAPAS----------------------')


        queries = []
        queries.append(inp_question)  # pegando apenas 1 pergunta

        table = corpus_tables[hits[0]["corpus_id"]]
        table = table.astype(str)
        start_time = time.time()
        inputs = reader_tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
        outputs = reader_model(**inputs)
        predicted_answer_coordinates, predicted_aggregation_indices = reader_tokenizer.convert_logits_to_predictions(
        inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())
        print(f'Reader TAPAS para top-{num_candidates} levou {(time.time() - start_time)} seconds')


        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
        aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

        answers = []
        for coordinates in predicted_answer_coordinates:
            if len(coordinates) == 1:
                print("Resposta obtida em only a single cell:")
                answers.append(corpus_tables[hits[0]["corpus_id"]].iat[coordinates[0]])
            else:
                print("Resposta obtida em multiple cells")
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(corpus_tables[hits[0]["corpus_id"]].iat[coordinate])
                answers.append(", ".join(cell_values))

        print("")
        for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
            print(query)
            #if predicted_agg == "NONE":
            #    print("Predicted answer: " + answer)
            #else:
            print("Predicted answer: " + predicted_agg + " aggregation > " + answer)
        print('')
        print(f'Tabela id: {hits[0]["corpus_id"]}')
        print('---------------------------------------') 

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
        embedding_path = "/data/ott-qa/"
        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        path_local = "/modelos/deepset_all-mpnet-base-v2-table" #deepset_all-mpnet-base-v2-tablel.pth"
        retriever_biencoder_model = SentenceTransformer(path_local, device=device)
    
        # Reader
        reader_model_name = "google/tapas-large-finetuned-wikisql-supervised"
        config = TapasConfig(reader_model_name)    # baixando o modelo do hugging face
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







