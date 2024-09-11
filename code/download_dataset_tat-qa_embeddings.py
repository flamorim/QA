# max seq len: Specifies the maximum sequence length of one input text for the model. Mandatory.

### versao com sentença e tabela nos embeddings

from sentence_transformers import SentenceTransformer, util
import tqdm, json, os
import pandas as pd
import pickle
import torch
import re

PATH_TO_JSON = '/data/ott-qa/traindev_tables.json'
debug_mode = True

def download_released_data_dir():
    files = ['tatqa_dataset_train.json','tatqa_dataset_dev.json','tatqa_dataset_test.json']
    for file in files:
        print(f'baixando {file}')
        url = "https://github.com/NExTplusplus/tat-qa/blob/master/dataset_raw/" + file
        local_file_name = "/data/tat-qa/released_data/" + file
        util.http_get(url, local_file_name)


def read_dataset_pickle():
    with open("/data/tat-qa/embeddings/new_dataset.pkl", "rb") as fIn:
            cache_data = pickle.load(fIn)
            tables_uid           = cache_data["tables_uid"]
            tables_ixd           = cache_data["tables_idx"]
            tables_body          = cache_data["tables_body"]
            tables_url           = cache_data["tables_url"]
            tables_title         = cache_data["tables_title"]
            tables_header        = cache_data["tables_header"]
            tables_section_title = cache_data["tables_section_title"]
            tables_section_text  = cache_data["tables_section_text"]
            tables_intro         = cache_data["tables_intro"]
    print(f'tamanho dataset: {len(tables_uid)}')
    print("")


def _preprocess_tables(tables: list):
    processed = []
    # loop through all tables
    for table in tables:
        organized_columns = [sanitized_column_name(col) for col in table.columns]
        table.columns = organized_columns

        #print(table)  #.replace("")
        # convert the table to csv and
        processed_table = "\n".join([table.to_csv(index=False)])
        # add the processed table to processed list
        processed.append(processed_table)

    return processed



def sanitized_column_name(col_name):

    return re.sub(r"\W+","-",col_name)


def create_csv_organized_col():

    with open(PATH_TO_JSON, 'r') as f:
        json_data = json.load(f)

    tables = []
    tables_title = []
    tables_sentence = []
    tables_url = []
    tables_uid = []
    count = 0
    for key,value in json_data.items():
        columns = [col[0] for col in value["header"]]

        organized_columns = [sanitized_column_name(col) for col in columns]

        data = []
        for entry in value["data"]:
            row = [item[0] for item in entry]
            data.append(row)

        # Criar DataFrame
        df = pd.DataFrame(data, columns=organized_columns)
        tables.append(df)

        csv_name = '/data/ott-qa/new_csv/organized_col/' + value["uid"].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.').replace('*', '')   + '.csv'
        df.to_csv(csv_name,index=False)
        print(csv_name)

        if ((count % 100) == 0):
            print(count)
        count +=1



###########################                    

# Function to create CSV file for each table
def create_csv_files_from_json(data):
    for item in data:
        table_data = item['table']['table']
        uid = item['table']['uid']

        # Create DataFrame from the table data
        df = pd.DataFrame(table_data[1:], columns=table_data[0])

        # Save DataFrame to a CSV file named by the uid
        csv_filename = f"/data/tat-qa/csv/{uid}.csv"
        df.to_csv(csv_filename, index=False)

# Function to create a single TXT file for each set of paragraphs
def create_txt_files_from_json(data):
    for item in data:
        paragraphs = item.get('paragraphs', [])
        uid = item['table']['uid']

        # Concatenate the text of all paragraphs
        combined_text = "\n".join(paragraph['text'] for paragraph in paragraphs)

        # Create a TXT file with the combined text
        txt_filename = f"/data/tat-qa/csv/{uid}.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(combined_text)

    print("TXT files created successfully.")

def create_dataset_json(data):
    # Prepare the data for the new JSON
        questions_data = []

        # Iterate through each item and extract question-related data
        for item in data:
            table_uid = item['table']['uid']
            questions = item.get('questions', [])
            
            for question in questions:
                question_info = {
                    'question_id': question['uid'],
                    'question':question['question'],
                    'table_id': table_uid,
                    'answer-text': question['answer'],
                    'answer_type': question['answer_type'],
                    'answer_from': question['answer_from']
                }
                if question['answer_from'] == 'table':
                    questions_data.append(question_info)

        # Save the new data to a JSON file
        output_filename = '/data/tat-qa/released_data/dev.json'
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, ensure_ascii=False, indent=4)

        print("JSON file created successfully.")
        print(len(questions_data))




def create_dataset():

    with open(PATH_TO_JSON, 'r') as f:    #traindev_tables.json
        json_data = json.load(f)

    tables_header = []
    tables_body   = []
    tables_title  = []
    tables_section_title = []
    tables_section_text  = []
    tables_intro  = []
    tables_url    = []
    tables_uid    = []
    tables_idx    = []
    count = 0
    for value in json_data:
        #columns = [col[0] for col in value["header"]]

        #data = []
        #for entry in value["data"]:
        #    row = [item[0] for item in entry]
        #    data.append(row)

        # Criar DataFrame
        table_id = value['table_id']
        table_id = '/data/tat-qa/csv/' + table_id + '.csv'
        if table_id in tables_uid:
            continue


                
        df = pd.read_csv(table_id,index_col=None)
        tables_body.append(df)
        passage_id =  value['table_id']
        passage_id = '/data/tat-qa/csv/' + passage_id + '.txt'
        with open(passage_id, 'r', encoding='utf-8') as file:
            intro = file.read()
        tables_intro.append(intro)

        title = 'n/a'
        tables_title.append(title)

        url = 'n/a'
        tables_url.append(url)

        header = 'n/a'
        tables_header.append(header)
        #tables_header.append(columns)


        section_title = 'n/a'
        tables_section_title.append(section_title)

        section_text =  'n/a'
        tables_section_text.append(section_text)

        #uid = value['uid'].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.').replace('*', '')
        tables_uid.append(table_id)

        tables_idx.append(count)
        if ((count % 100) == 0):
            print(count)
        count +=1

    with open("/data/tat-qa/embeddings/new_dataset.pkl", "wb") as fOut:
        pickle.dump({"tables_uid": tables_uid,
                     "tables_idx": tables_idx,
                     "tables_body": tables_body,
                     "tables_url": tables_url,
                     "tables_title": tables_title,
                     "tables_header": tables_header,
                     "tables_section_title": tables_section_title,
                     "tables_section_text": tables_section_text,
                     "tables_intro": tables_intro}, fOut)

    print(f'criado pkl com {len(tables_idx)} elementos')

def create_mpnet_embeddings_table(device,max_sequency):
        corpus = []
        print('criando embedding das tabelas')
        with open("/data/tat-qa/embeddings/new_dataset.pkl", "rb") as fIn:
            cache_data = pickle.load(fIn)
            # lendo o dataset
            tables_idx           = cache_data["tables_idx"]
            tables_uid           = cache_data["tables_uid"]
            tables_body          = cache_data["tables_body"]
            tables_url           = cache_data["tables_url"]
            tables_title         = cache_data["tables_title"]
            tables_header        = cache_data["tables_header"]
            tables_section_title = cache_data["tables_section_title"]
            tables_section_text  = cache_data["tables_section_text"]
            tables_intro         = cache_data["tables_intro"]

        processed_tables = _preprocess_tables(tables_body)

        ###processed_tables = processed_tables[0:10] ################

        #for (sentence,table) in list(zip(tables_sentence, processed_tables)):
        #    doc = sentence + '\n' + table
        #    corpus.append(doc)

        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        retriever_biencoder_model = SentenceTransformer(retriever_model_name, device=device)

        # that's the sentence transformer como ele foi treinado = 384
        retriever_biencoder_model.max_seq_length = max_sequency
        max_seq = retriever_biencoder_model.max_seq_length
        
        # that's the underlying transformer o quanto ele pode ter = 512
        max_pos = retriever_biencoder_model[0].auto_model.config.max_position_embeddings = 512

        corpus_embeddings = retriever_biencoder_model.encode(processed_tables, convert_to_tensor=True, show_progress_bar=True)
        print(f'Salvando localmente os embeddings feitos em {device}')

        input_ids = retriever_biencoder_model.tokenizer(processed_tables)['input_ids']
        tables_tokens_len = []
        for input_id in input_ids:
            num_tokens = len(input_id)
            tables_tokens_len.append(num_tokens)


        # fazendo com os nomes de colunas melhorado e salvando os novos embeddings
        file_name = f'/data/tat-qa/embeddings/mpnet_table_embeddings_{device}_{max_seq}_{max_pos}.pkl'

        with open(file_name, "wb") as fOut:
            #pickle.dump({"tables": tables, "embeddings": corpus_embeddings, "tables_url": tables_url, "tables_sentence": tables_sentence, "tables_title": tables_title}, fOut)

            pickle.dump({
                "embeddings": corpus_embeddings,
                "tables_idx": tables_idx,
                "tables_uid": tables_uid,
                "tables_body": tables_body,
                "tables_url": tables_url,
                "tables_title": tables_title,
                "tables_header": tables_header,
                "tables_section_title": tables_section_title,
                "tables_section_text": tables_section_text,
                "tables_intro": tables_intro,
                "embedding_tokens_len": tables_tokens_len}, fOut)
#                "tables_and_append_tokens_len":tables_tokens_len}, fOut)  # só tenho a tabela


        print(f'criado embedds com {len(corpus_embeddings)} vetores')



def create_mpnet_embeddings_table_intro(device,max_sequency):
        corpus = []
        print('criando embedding das tabelas')
        with open("/data/tat-qa/embeddings/new_dataset.pkl", "rb") as fIn:
            cache_data = pickle.load(fIn)
            # lendo o dataset
            tables_idx           = cache_data["tables_idx"]
            tables_uid           = cache_data["tables_uid"]
            tables_body          = cache_data["tables_body"]
            tables_url           = cache_data["tables_url"]
            tables_title         = cache_data["tables_title"]
            tables_header        = cache_data["tables_header"]
            tables_section_title = cache_data["tables_section_title"]
            tables_section_text  = cache_data["tables_section_text"]
            tables_intro         = cache_data["tables_intro"]

        processed_tables = _preprocess_tables(tables_body)

        ###processed_tables = processed_tables[0:10] ################

        for (sentence,table) in list(zip(tables_intro, processed_tables)):
            doc = sentence + '\n' + table
            corpus.append(doc)

        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        retriever_biencoder_model = SentenceTransformer(retriever_model_name, device=device)

        # that's the sentence transformer como ele foi treinado = 384
        retriever_biencoder_model.max_seq_length = max_sequency
        max_seq = retriever_biencoder_model.max_seq_length
        
        # that's the underlying transformer o quanto ele pode ter = 512
        max_pos = retriever_biencoder_model[0].auto_model.config.max_position_embeddings = 512

        corpus_embeddings = retriever_biencoder_model.encode(processed_tables, convert_to_tensor=True, show_progress_bar=True)
        print(f'Salvando localmente os embeddings feitos em {device}')

        input_ids = retriever_biencoder_model.tokenizer(processed_tables)['input_ids']
        tables_tokens_len = []
        for input_id in input_ids:
            num_tokens = len(input_id)
            tables_tokens_len.append(num_tokens)


        # fazendo com os nomes de colunas melhorado e salvando os novos embeddings
        file_name = f'/data/tat-qa/embeddings/mpnet_table_intro_embeddings_{device}_{max_seq}_{max_pos}.pkl'

        with open(file_name, "wb") as fOut:
            #pickle.dump({"tables": tables, "embeddings": corpus_embeddings, "tables_url": tables_url, "tables_sentence": tables_sentence, "tables_title": tables_title}, fOut)

            pickle.dump({
                "embeddings": corpus_embeddings,
                "tables_idx": tables_idx,
                "tables_uid": tables_uid,
                "tables_body": tables_body,
                "tables_url": tables_url,
                "tables_title": tables_title,
                "tables_header": tables_header,
                "tables_section_title": tables_section_title,
                "tables_section_text": tables_section_text,
                "tables_intro": tables_intro,
                "embedding_tokens_len": tables_tokens_len}, fOut)
#                "tables_and_append_tokens_len":tables_tokens_len}, fOut)  # só tenho a tabela

        print(f'criado embedds com {len(corpus_embeddings)} vetores')

def ajusta_dataset_pickle():
    with open("/data/tat-qa/embeddings/mpnet_table_embeddings_cpu_512_512.pkl", "rb") as fIn:
            cache_data = pickle.load(fIn)
            tables_uid           = cache_data["tables_uid"]
            tables_ixd           = cache_data["tables_idx"]
            tables_body          = cache_data["tables_body"]
            tables_url           = cache_data["tables_url"]
            tables_title         = cache_data["tables_title"]
            tables_header        = cache_data["tables_header"]
            tables_section_title = cache_data["tables_section_title"]
            tables_section_text  = cache_data["tables_section_text"]
            tables_intro         = cache_data["tables_intro"]
    print(f'tamanho dataset: {len(tables_uid)}')
    print("")
    df = pd.DataFrame(tables_intro)
    df.to_csv(csv_filename, index=False)

debug_mode = True
if __name__ == '__main__':
    if debug_mode == True:
        import debugpy
        debugpy.listen(7011)
        print("Waiting for debugger attach")
        debugpy.wait_for_client() 
    print('hello 2')
    i = 1
    print('hello 3')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #device = torch.device("cpu") obrigando CPU
    print(device)


    # Parse JSON data
    json_file = '/data/tat-qa/tatqa_dataset_train.json'
    with open(json_file, 'r') as fin:
        data = json.load(fin)
    create_csv_files_from_json(data)
    # Create TXT files
    #create_txt_files_from_json(data)


    #PATH_TO_JSON = '/data/tat-qa/released_data/dev.json'
    #create_dataset()  # gera o pkl

    #create_mpnet_embeddings_table('cpu',512) # transforma o pkl em embeddings
    #create_mpnet_embeddings_table_intro('cpu',512) # transforma o pkl em embeddings
    #ajusta_dataset_pickle()
    
    print('fim')
    
#create_csv_organized_col()
