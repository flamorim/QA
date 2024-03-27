import torch
import os
import pickle
import time
#para o fine tunning
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score 
import torch 
from transformers import TrainingArguments, Trainer 
from transformers import BertTokenizer, BertForSequenceClassification
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

#apid.petrobras.com.br
#APID:6a1b62decb914f4291b754bea0f92d21
#apit.petrobras.com.br
#APIT:239042549f214f75abc4a006034eb4d0
#api.petrobras.com.br
#API:72b26ee264b5440ca36cdf717ee80712

##os.environ["OPENAI_API_KEY"] = '72b26ee264b5440ca36cdf717ee80712'
##os.environ["OPENAI_API_BASE"] = 'https://api.petrobras.com.br'
##os.environ["OPENAI_API_VERSION"] = '2023-03-15-preview'
##os.environ["OPENAI_API_TYPE"] = 'azure'
##os.environ["REQUESTS_CA_BUNDLE"] = "/nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/petrobras-openai/petrobras-ca-root.pem"
##APIResource.azure_api_prefix = 'ia/openai/v1/openai-azure/openai'
##print(os.environ["REQUESTS_CA_BUNDLE"])

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
####################################funcoes###########################
# Create torch dataset 
class Dataset(torch.utils.data.Dataset): 
    def __init__(self, encodings, labels=None): 
        self.encodings = encodings 
        self.labels = labels

    def __getitem__(self, idx): 
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} 
        if self.labels: 
            item["labels"] = torch.tensor(self.labels[idx]) 
        return item
        
    def __len__(self): 
        return len(self.encodings["input_ids"])

def compute_metrics(p):
    print(type(p)) 
    pred, labels = p 
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

####################################funcoes###########################


def main():
    data = pd.read_csv("/data/toxic/train.csv",error_bad_lines=False, engine="python") 
    print(data.head())

    print(data['toxic'].value_counts())
    data = data[['comment_text','toxic']] 
    data = data[0:1000] 
    print(data.head())
    print(data['toxic'].value_counts())

    from sklearn.model_selection import train_test_split 
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score 

    from transformers import TrainingArguments, Trainer 
    from transformers import BertTokenizer, BertForSequenceClassification


    from transformers import BertTokenizer, BertForSequenceClassification 
    diretorio_tokenizer = "/modelos/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(diretorio_tokenizer)


    # Carregar o modelo
    diretorio_modelo = "/modelos/bert-base-uncased-ForSequenceClassification-2-classes"
    model = BertForSequenceClassification.from_pretrained(diretorio_modelo,num_labels=2)

    sample_data = ["I am eating","I am playing "] 
    tokenizer(sample_data, padding=True, truncation=True, max_length=512)

    X = list(data["comment_text"]) 
    y = list(data["toxic"]) 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,stratify=y) 
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512) 
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

    X_train_tokenized.keys()

    #print(X_train_tokenized['attention_mask'][0])
    #print(len(X_train),len(X_val))

    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    #print(train_dataset[5])
    # Define Trainer 
    args = TrainingArguments(
        output_dir="/modelos",
        num_train_epochs=1,
        per_device_train_batch_size=8 ) 
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics)
        
    #trainer.train()
    #trainer.save_model('/modelos/CustomModel/toxic_classification')
    model = BertForSequenceClassification.from_pretrained("/modelos/CustomModel/toxic_classification") 
    #model.to('cuda')


    #trainer.evaluate()

    np.set_printoptions(suppress=True)

    text = "That was good point" 
    # text = "go to hell" 
    inputs = tokenizer(text,padding = True, truncation = True, return_tensors='pt') #.to('cuda') 
    outputs = model(**inputs) 
    print(outputs) 
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1) 
    print(predictions) 
    predictions = predictions.detach().numpy() #predictions.cpu().detach().numpy() 
    print(predictions)

    text = "Está quente como no inferno"
    inputs = tokenizer(text,padding = True, truncation = True, return_tensors='pt') #.to('cuda') 
    outputs = model(**inputs) 
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1) 
    predictions = predictions.detach().numpy()  #predictions.cpu().detach().numpy()
    print(predictions)


debug_mode = True
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
    main()


