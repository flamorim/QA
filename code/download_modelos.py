#from tabfact import get_embeddings
import torch
import os
import pickle
import time
# retriever
from sentence_transformers import SentenceTransformer, util


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
device = torch.device("cpu") ##obrigando CPU
print(device)


debug_mode = False
if __name__ == '__main__':
    if debug_mode == True:
        import debugpy
        debugpy.listen(7011)
        print("Waiting for debugger attach")
        debugpy.wait_for_client() 
    print('hello 2')
    i = 1
    print('hello 3')


def salva_modelo():
    retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
    retriever_biencoder_model = SentenceTransformer(retriever_model_name, device=device)
    #tokenizador = AutoTokenizer.from_pretrained(modelo_nome)

    # Diretório para salvar o modelo
    diretorio_destino = "/QA/Bert/data/tabfact/deepset_all-mpnet-base-v2-table" #deepset_all-mpnet-base-v2-tablel.pth"
    # Salvar o modelo e o tokenizador no diretório especificado
    # Salvar o modelo
    retriever_biencoder_model.save(diretorio_destino)

def carrega_modelo():
    diretorio_destino = "/QA/Bert/data/tabfact/deepset_all-mpnet-base-v2-table" #deepset_all-mpnet-base-v2-tablel.pth"
#    retriever_biencoder_model = SentenceTransformer(retriever_model_name, device=device)
    retriever_biencoder_model = SentenceTransformer(diretorio_destino)

def salva_tapas():
    from transformers import TapasConfig
    from transformers import TapasTokenizer, TapasForQuestionAnswering
    reader_model_name = "google/tapas-large-finetuned-wikisql-supervised"
    config = TapasConfig(reader_model_name)
    reader_model = TapasForQuestionAnswering.from_pretrained(reader_model_name)
    reader_tokenizer = TapasTokenizer.from_pretrained(reader_model_name)

    diretorio_destino = "/modelos/google_model_tapas-large-finetuned-wikisql-supervised"
    reader_model.save_pretrained(diretorio_destino)
    #reader_model.save_model("diretorio_destino")
    diretorio_destino = "/modelos/google_tokenizer_tapas-large-finetuned-wikisql-supervised"
    reader_tokenizer.save_pretrained(diretorio_destino)
    #reader_tokenizer.save_model("diretorio_destino")

def carrega_tapas():
    from transformers import TapasTokenizer, TapasForQuestionAnswering
    # Carregar o tokenizer
    diretorio_tokenizer = "/modelos/google_tokenizer_tapas-large-finetuned-wikisql-supervised"
    reader_tokenizer = TapasTokenizer.from_pretrained(diretorio_tokenizer)

    # Carregar o modelo
    diretorio_modelo = "/modelos/google_model_tapas-large-finetuned-wikisql-supervised"
    reader_model = TapasForQuestionAnswering.from_pretrained(diretorio_modelo)

def salva_bert_2_classes():
    from transformers import BertTokenizer, BertForSequenceClassification 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
    diretorio_destino = "/modelos/bert-base-uncased"
    tokenizer.save_pretrained(diretorio_destino)


    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
    diretorio_destino = "/modelos/bert-base-uncased-ForSequenceClassification-2-classes"
    model.save_pretrained(diretorio_destino)

def carrega_bert_2_classes():
    from transformers import BertTokenizer, BertForSequenceClassification 
    # Carregar o tokenizer
    diretorio_tokenizer = "/modelos/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(diretorio_tokenizer)
    print(tokenizer)

    # Carregar o modelo
    diretorio_modelo = "/modelos/bert-base-uncased-ForSequenceClassification-2-classes"
    model = BertForSequenceClassification.from_pretrained(diretorio_modelo,num_labels=2)
    print(model)


def salva_tapex():
    from transformers import TapexTokenizer, BartForConditionalGeneration

    tapex_model_name = "microsoft/tapex-large-finetuned-wtq"
    tapex_tokenizer = TapexTokenizer.from_pretrained(tapex_model_name)
    tapex_model = BartForConditionalGeneration.from_pretrained(tapex_model_name)
    diretorio_destino = "/modelos/microsoft_tapex-large-finetuned-wtq"

    #tapex_tokenizer.save_pretrained(diretorio_destino)
    #tapex_model.save_pretrained(diretorio_destino)

def carrega_tapex():
    from transformers import TapexTokenizer, BartForConditionalGeneration

    diretorio = "/modelos/microsoft_tapex-large-finetuned-wtq"
    tapex_tokenizer = TapexTokenizer.from_pretrained(diretorio)
    print(tapex_tokenizer)
    
    tapex_model = BartForConditionalGeneration.from_pretrained(diretorio)
    print(tapex_model)

def salva_mdr():
    #from transformers import AutoTokenizer
    #from transformers import RobertaEncoder # MDREncoder
    
    # Load model directly
    from transformers import AutoTokenizer, RobertaEncoder

    tokenizer = AutoTokenizer.from_pretrained("kiyoung2/dpr_q-encoder_roberta-small")
    model = RobertaEncoder.from_pretrained("kiyoung2/dpr_q-encoder_roberta-small")




    mdr_model_name = "deutschmann/mdr_roberta_q_encoder"

    tokenizer = AutoTokenizer.from_pretrained(mdr_model_name)
    model = MDREncoder.from_pretrained(mdr_model_name)

    diretorio_destino = "/modelos/deutschmann_mdr_roberta_q_encoder"

    tokenizer.save_pretrained(diretorio_destino)
    model.save_pretrained(diretorio_destino)

def salva_reranker():
    from sentence_transformers import CrossEncoder
    rerank_cross_encoder_model = CrossEncoder("cross-encoder/stsb-roberta-base")
    #The model will predict scores for the pairs ('Sentence 1', 'Sentence 2')
    diretorio_destino = "/modelos/reranker/cross-encoder-stsb-roberta-base"
    rerank_cross_encoder_model.save_pretrained(diretorio_destino)

def carrega_reranker():
    from sentence_transformers import CrossEncoder
    #The model will predict scores for the pairs ('Sentence 1', 'Sentence 2')
    diretorio = "/modelos/reranker/cross-encoder-stsb-roberta-base"
    rerank_cross_encoder_model = CrossEncoder(diretorio)
    print(rerank_cross_encoder_model)

def salva_reranker_marco_MiniLM():
    from sentence_transformers import CrossEncoder
    rerank_cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    #The model will predict scores for the pairs ('Sentence 1', 'Sentence 2')
    diretorio_destino = "/modelos/reranker/cross-encoder-ms-marco-MiniLM-L-6-v2"
    rerank_cross_encoder_model.save_pretrained(diretorio_destino)

def carrega_reranker_marco_MiniLM():
    from sentence_transformers import CrossEncoder
    #The model will predict scores for the pairs ('Sentence 1', 'Sentence 2')
    diretorio = "/modelos/reranker/cross-encoder-ms-marco-MiniLM-L-6-v2"
    rerank_cross_encoder_model = CrossEncoder(diretorio)
    print(rerank_cross_encoder_model)




#sbatch Start-rr-llm.srm

#salva_modelo()
#carrega_modelo()
#salva_tapas()
#carrega_tapas()
#salva_bert_2_classes()
#carrega_bert_2_classes()
#salva_tapex()
#carrega_tapex()
#salva_mdr()
#carrega_reranker()
#salva_reranker_marco_MiniLM()
#carrega_reranker_marco_MiniLM()


diretorio = "/QA/Bert/modelos/deepset_all-mpnet-base-v2-table"
model = SentenceTransformer(diretorio)



from sentence_transformers import CrossEncoder
diretorio = "/modelos/reranker/cross-encoder-ms-marco-MiniLM-L-6-v2"
model = CrossEncoder(diretorio)

#num_layers = len(retriever_biencoder_model.encoder.layer)
num_layers = len(model.bert.encoder.layer)
print(num_layers)

total_params = sum(p.numel() for p in model.parameters())

#print(model)
print(total_params)   

# retriever 109.486.464
