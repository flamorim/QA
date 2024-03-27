# max seq len: Specifies the maximum sequence length of one input text for the model. Mandatory.

### versao com sentença e tabela nos embeddings

from sentence_transformers import SentenceTransformer, util
import tqdm, json, os
import pandas as pd
import pickle
import torch

PATH_TO_JSON = '/data/ott-qa/traindev_tables.json'
debug_mode = True


def download_released_data_dir():
    files = ['dev.json','dev.traced.json','dev_reference.json','test.blind.json','train.json','train.traced.json','train_dev_test_table_ids.json']
    for file in files:
        print(f'baixando {file}')
        url = "https://github.com/wenhuchen/OTT-QA/tree/master/released_data/" + file
        local_file_name = "/data/ott-qa/released_data/" + file
        util.http_get(url, local_file_name)


def download_bootstrap():
    "arquivo com metadados"
    url = "https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/master/bootstrap/bootstrap.json"
    local_file_name = "/home/Bert/data/tabfact/bootstrap.json"
    util.http_get(url, local_file_name)



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
    for key,value in json_data.items():
        columns = [col[0] for col in value["header"]]

        data = []
        for entry in value["data"]:
            row = [item[0] for item in entry]
            data.append(row)

        # Criar DataFrame
        df = pd.DataFrame(data, columns=columns)
        tables_body.append(df)

        intro = value["intro"]
        tables_intro.append(intro)

        title = value["title"]
        tables_title.append(title)

        url = value["url"]
        tables_url.append(url)

        header = value['header']
        columns = [col[0] for col in value["header"]]
        header = ', '.join(columns)
        tables_header.append(header)

        section_title = value['section_title']
        tables_section_title.append(section_title)

        section_text = value['section_text']
        tables_section_text.append(section_text)

        uid = value['uid'].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.')
        tables_uid.append(uid)

        tables_idx.append(count)
        if ((count % 100) == 0):
            print(count)
        count +=1

    with open("/data/ott-qa/embeddings/new_dataset.pkl", "wb") as fOut:
        pickle.dump({"tables_uid": tables_uid,
                     "tables_idx": tables_idx,
                     "tables_body": tables_body,
                     "tables_url": tables_url,
                     "tables_title": tables_title,
                     "tables_header": tables_header,
                     "tables_section_title": tables_section_title,
                     "tables_section_text": tables_section_text,
                     "tables_intro": tables_intro}, fOut)

def read_dataset_pickle():
    with open("/data/ott-qa/embeddings/new_dataset.pkl", "rb") as fIn:
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
    count = 0
    for table in tables:
        # convert the table to csv and
        processed_table = "\n".join([table.to_csv(index=False)])
        # add the processed table to processed list
        processed.append(processed_table)
        #if count == 10:
        #    break
        #count +=1
    return processed

def create_mpnet_embeddings_table_intro(device):
        corpus = []
        print('criando embedding das tabelas com passages')
        with open("/data/ott-qa/embeddings/new_dataset.pkl", "rb") as fIn:
            cache_data = pickle.load(fIn)
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

        for (sentence,table) in list(zip(tables_intro, processed_tables)):
            doc = sentence + '\n' + table
            corpus.append(doc)

        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        retriever_biencoder_model = SentenceTransformer(retriever_model_name, device=device)

        # that's the sentence transformer como ele foi treinado = 384
        retriever_biencoder_model.max_seq_length = 512
        max_seq = retriever_biencoder_model.max_seq_length
        
        # that's the underlying transformer o quanto ele pode ter = 514
        max_pos = retriever_biencoder_model[0].auto_model.config.max_position_embeddings

        corpus_embeddings = retriever_biencoder_model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        print(f'Salvando localmente os embeddings feitos em {device}')

        input_ids = retriever_biencoder_model.tokenizer(processed_tables)['input_ids']
        tables_tokens_len = []
        for input_id in input_ids:
            num_tokens = len(input_id)
            tables_tokens_len.append(num_tokens)

        input_ids = retriever_biencoder_model.tokenizer(corpus)['input_ids']
        tables_and_append_tokens_len = []
        for input_id in input_ids:
            num_tokens = len(input_id)
            tables_and_append_tokens_len.append(num_tokens)

        file_name = f'/data/ott-qa/embeddings/mpnet_table_intro_embeddings_{device}_{max_seq}_{max_pos}.pkl'
        with open(file_name, "wb") as fOut:
            #pickle.dump({"tables": tables, "embeddings": corpus_embeddings, "tables_url": tables_url, "tables_sentence": tables_sentence, "tables_title": tables_title}, fOut)
            pickle.dump({
                "embeddings": corpus_embeddings,
                "tables_idx" : tables_idx,
                "tables_uid": tables_uid,
                "tables_body": tables_body,
                "tables_url": tables_url,
                "tables_title": tables_title,
                "tables_header": tables_header,
                "tables_section_title": tables_section_title,
                "tables_section_text": tables_section_text,
                "tables_intro": tables_intro,
                "tables_tokens_len": tables_tokens_len,
                "tables_and_append_tokens_len": tables_and_append_tokens_len}, fOut)


def create_mpnet_embeddings_table_header(device):
        corpus = []
        print('criando embedding das tabelas com passages')
        with open("/data/ott-qa/embeddings/new_dataset.pkl", "rb") as fIn:
            cache_data = pickle.load(fIn)
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

        for (sentence,table) in list(zip(tables_header, processed_tables)):
            doc = sentence + '\n' + table
            corpus.append(doc)

        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        retriever_biencoder_model = SentenceTransformer(retriever_model_name, device=device)

        # that's the sentence transformer como ele foi treinado = 384
        retriever_biencoder_model.max_seq_length = 512
        max_seq = retriever_biencoder_model.max_seq_length
        
        # that's the underlying transformer o quanto ele pode ter = 512
        max_pos = retriever_biencoder_model[0].auto_model.config.max_position_embeddings

        corpus_embeddings = retriever_biencoder_model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        print(f'Salvando localmente os embeddings feitos em {device}')

        input_ids = retriever_biencoder_model.tokenizer(processed_tables)['input_ids']
        tables_tokens_len = []
        for input_id in input_ids:
            num_tokens = len(input_id)
            tables_tokens_len.append(num_tokens)

        input_ids = retriever_biencoder_model.tokenizer(corpus)['input_ids']
        tables_and_append_tokens_len = []
        for input_id in input_ids:
            num_tokens = len(input_id)
            tables_and_append_tokens_len.append(num_tokens)


        file_name = f'/data/ott-qa/embeddings/mpnet_table_header_embeddings_{device}_{max_seq}_{max_pos}.pkl'
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
                "tables_tokens_len": tables_tokens_len,
                "tables_and_append_tokens_len": tables_and_append_tokens_len}, fOut)


def create_mpnet_embeddings_table_section_text(device):
        corpus = []
        print('criando embedding das tabelas com passages')
        with open("/data/ott-qa/embeddings/new_dataset.pkl", "rb") as fIn:
            cache_data = pickle.load(fIn)
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

        for (sentence,table) in list(zip(tables_section_text, processed_tables)):
            doc = sentence + '\n' + table
            corpus.append(doc)

        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        retriever_biencoder_model = SentenceTransformer(retriever_model_name, device=device)

        # that's the sentence transformer como ele foi treinado = 384
        retriever_biencoder_model.max_seq_length = 512
        max_seq = retriever_biencoder_model.max_seq_length
        
        # that's the underlying transformer o quanto ele pode ter = 512
        max_pos = retriever_biencoder_model[0].auto_model.config.max_position_embeddings

        corpus_embeddings = retriever_biencoder_model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        print(f'Salvando localmente os embeddings feitos em {device}')

        input_ids = retriever_biencoder_model.tokenizer(processed_tables)['input_ids']
        tables_tokens_len = []
        for input_id in input_ids:
            num_tokens = len(input_id)
            tables_tokens_len.append(num_tokens)

        input_ids = retriever_biencoder_model.tokenizer(corpus)['input_ids']
        tables_and_append_tokens_len = []
        for input_id in input_ids:
            num_tokens = len(input_id)
            tables_and_append_tokens_len.append(num_tokens)

        file_name = f'/data/ott-qa/embeddings/mpnet_table_section_text_embeddings_{device}_{max_seq}_{max_pos}.pkl'
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
                "tables_tokens_len": tables_tokens_len,
                "tables_and_append_tokens_len": tables_and_append_tokens_len}, fOut)

def create_mpnet_embeddings_table_section_title(device):
        corpus = []
        print('criando embedding das tabelas com passages')
        with open("/data/ott-qa/embeddings/new_dataset.pkl", "rb") as fIn:
            cache_data = pickle.load(fIn)
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

        for (sentence,table) in list(zip(tables_section_title, processed_tables)):
            doc = sentence + '\n' + table
            corpus.append(doc)

        retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        retriever_biencoder_model = SentenceTransformer(retriever_model_name, device=device)

        # that's the sentence transformer como ele foi treinado = 384
        retriever_biencoder_model.max_seq_length = 512
        max_seq = retriever_biencoder_model.max_seq_length
        
        # that's the underlying transformer o quanto ele pode ter = 512
        max_pos = retriever_biencoder_model[0].auto_model.config.max_position_embeddings
        #corpus = corpus[0:10]
        corpus_embeddings = retriever_biencoder_model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        print(f'Salvando localmente os embeddings feitos em {device}')

        input_ids = retriever_biencoder_model.tokenizer(processed_tables)['input_ids']
        tables_tokens_len = []
        for input_id in input_ids:
            num_tokens = len(input_id)
            tables_tokens_len.append(num_tokens)
  
        input_ids = retriever_biencoder_model.tokenizer(corpus)['input_ids']
        tables_and_append_tokens_len = []
        for input_id in input_ids:
            num_tokens = len(input_id)
            tables_and_append_tokens_len.append(num_tokens)



        file_name = f'/data/ott-qa/embeddings/mpnet_table_section_title_embeddings_{device}_{max_seq}_{max_pos}.pkl'
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
                "tables_tokens_len": tables_tokens_len,
                "tables_and_append_tokens_len": tables_and_append_tokens_len}, fOut)


def create_mpnet_embeddings_table(device):
        corpus = []
        print('criando embedding das tabelas')
        with open("/data/ott-qa/embeddings/new_dataset.pkl", "rb") as fIn:
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
        retriever_biencoder_model.max_seq_length = 512
        max_seq = retriever_biencoder_model.max_seq_length
        
        # that's the underlying transformer o quanto ele pode ter = 512
        max_pos = retriever_biencoder_model[0].auto_model.config.max_position_embeddings

        corpus_embeddings = retriever_biencoder_model.encode(processed_tables, convert_to_tensor=True, show_progress_bar=True)
        print(f'Salvando localmente os embeddings feitos em {device}')

        input_ids = retriever_biencoder_model.tokenizer(processed_tables)['input_ids']
        tables_tokens_len = []
        for input_id in input_ids:
            num_tokens = len(input_id)
            tables_tokens_len.append(num_tokens)
        file_name = f'/data/ott-qa/embeddings/mpnet_table_embeddings_{device}_{max_seq}_{max_pos}.pkl'

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
                "tables_tokens_len": tables_tokens_len,
                "tables_and_append_tokens_len":tables_tokens_len}, fOut)  # só tenho a tabela


###################### TAPEX ####################
def create_tapex_embeddings_table(device):
        from transformers import TapexTokenizer, BartForConditionalGeneration

        corpus = []
        print('criando embedding das tabelas')
        with open("/data/ott-qa/embeddings/new_dataset.pkl", "rb") as fIn:
            cache_data = pickle.load(fIn)

            tables_uid           = cache_data["tables_uid"]
            tables_body          = cache_data["tables_body"]
            tables_url           = cache_data["tables_url"]
            tables_title         = cache_data["tables_title"]
            tables_header        = cache_data["tables_header"]
            tables_section_title = cache_data["tables_section_title"]
            tables_section_text  = cache_data["tables_section_text"]
            tables_intro         = cache_data["tables_intro"]

        processed_tables = _preprocess_tables(tables_body)

        # Load model directly
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
        model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large-finetuned-wtq")


        #diretorio = "/modelos/microsoft_tapex-large-finetuned-wtq"
        #retriever_biencoder_model = BartForConditionalGeneration.from_pretrained(diretorio)
        #retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
        #retriever_biencoder_model = SentenceTransformer(retriever_model_name, device=device)

#        corpus_embeddings = model.encode(processed_tables, convert_to_tensor=True, show_progress_bar=True)
        print(f'Salvando localmente os embeddings feitos em {device}')
        with open(f'/data/ott-qa/embeddings/tapex_table_embeddings_{device}.pkl', "wb") as fOut:
            #pickle.dump({"tables": tables, "embeddings": corpus_embeddings, "tables_url": tables_url, "tables_sentence": tables_sentence, "tables_title": tables_title}, fOut)

            pickle.dump({
                "embeddings": corpus_embeddings,
                "tables_uid": tables_uid,
                "tables_body": tables_body,
                "tables_url": tables_url,
                "tables_title": tables_title,
                "tables_header": tables_header,
                "tables_section_title": tables_section_title,
                "tables_section_text": tables_section_text,
                "tables_intro": tables_intro,
                "tables_tokens_len": tables_tokens_len}, fOut)

####################### fim #####################
def verifica_dataset():
    count = 0
    with open('/data/ott-qa/released_data/dev.json', 'r') as f:
        data = json.load(f)
    print(len(data))
    for qa in data:

        #inp_question = qa['question']
        answer_table_id = qa['table_id']  # ground trhuth
        csv_name = '/data/ott-qa/new_csv/' + answer_table_id.replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.')   + '.csv'

        if os.path.exists(csv_name):
            #print("ok")
            count += 1
        else:
            print(csv_name)
    print(count)


def create_csv():

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

        data = []
        for entry in value["data"]:
            row = [item[0] for item in entry]
            data.append(row)

        # Criar DataFrame
        df = pd.DataFrame(data, columns=columns)
        tables.append(df)

        csv_name = '/data/ott-qa/new_csv/' + value["uid"].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.')   + '.csv'
        df.to_csv(csv_name,index=False)
        print(csv_name)

        if ((count % 100) == 0):
            print(count)
        count +=1

def create_question_answer():


    with open('/data/ott-qa/released_data/dev.json', 'r') as f:    #traindev_tables.json
        json_data = json.load(f)

    questions_id     = []
    questions        = []
    tables_id        = []
    answers_text     = []
    questions_postag = []
    count = 0
    for key,value in json_data.items():
        print(key)
        #print(value)
        continue
        break
        question_id     = value["question_id"]
        questions_id.append(question_id)

        question        = value["question"]
        questions.append(question)

        table_id       = value["table_id"].replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.')
        tables_id.append(table__id)
        
        question_postag = value["question_postag"]
        questions_postag.append(question_postag)

        answer_text     = value["answer_text"]
        answers_text.append(answer_text)

        if ((count % 100) == 0):
            print(count)
        count +=1

    with open("/data/ott-qa/released_data/question_answer.pkl", "wb") as fOut:
        pickle.dump({"questions_id": questions_id,
                     "questions": questions,
                     "tables_id": tables_id,
                     "questions_postag": questions_postag,
                     "answers_text": answers_text}, fOut)


###########################                    




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

    #create_dataset()
    #read_dataset_pickle()
    create_mpnet_embeddings_table("cpu")
    #create_mpnet_embeddings_table_intro("cpu")
    #create_mpnet_embeddings_table_header("cpu")
    #create_mpnet_embeddings_table_section_text("cpu")
    #create_mpnet_embeddings_table_section_title("cpu")
    
    #create_csv()
    #download_released_data_dir()
    #create_question_answer()
    #verifica_dataset()
    #create_tapex_embeddings_table("cpu")

#download_all_csv()
#download_bootstrap()
#read_bootstrap()
#create_pickle()