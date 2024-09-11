import torch
import os
import pickle
import time, random
import json
import utils
import pandas as pd
import string
import re
import collections

import utils

from sentence_transformers import SentenceTransformer, util

#reader TAPAS
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering
from transformers import TapasConfig, AutoConfig

#reader TAPEX
from transformers import TapexTokenizer, BartForConditionalGeneration
#AutoModelForSeq2SeqLM:
#Esta classe é uma interface genérica que pode ser usada com qualquer modelo seq2seq de linguagem.


retriever_model_name = "deepset/all-mpnet-base-v2-table"  # para usar no nome dos embeddings salvos localmente
path_local = "/modelos/deepset_all-mpnet-base-v2-table" #deepset_all-mpnet-base-v2-tablel.pth"
device = "cpu"
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


def build_reader(retriever_file, device, run_improve_question):

            with open('/QA/Bert/code/path_to_files.json', 'r') as file:
                    path_to_files = json.load(file)

            dataset = pd.read_csv(retriever_file, sep=',') 
            #dataset = dataset.drop(dataset.index[50:0])  ## teste 

            df = pd.DataFrame()
            count_EM_tapex = 0
            count_EM_tapas = 0

            if run_improve_question == True:
                df_teste = pd.DataFrame()
                df_teste = pd.read_csv('/data/tat-qa/question_rewriting/improved_questions.csv',sep=',')
                questions_opt = df_teste['questions_opt'].to_list()




            for index, data in dataset.iterrows():
                if run_improve_question == True:
                    inp_question   = questions_opt[index]
                else:    
                    inp_question   = data['question_txt']
                # pegando a saída do retriever, isto é APENAS o primeiro da top100 dele

                ### atencao para a diferenca de pegar a saída dp retriever e a saída do reranking
                #answer_table_uid_list = eval(data['top100_table_uid'])

                ### atencao para a diferenca de pegar a saída dp retriever e a saída do reranking
                answer_table_uid_list = eval(data['rr_top100_table_uid'])
                
                tapex_answer_text_list = []
                tapas_answer_text_list = []
                aggregation_predictions_string_list = []

                print()
                print(f'linha {index}')
                print()
                ## atencao
                for answer_table_uid in answer_table_uid_list[0:10]:

                    table = answer_table_uid = answer_table_uid.replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace(',', '.').replace('*', '')  
                    # retirado pois tat-qa ja em o path completo
                    #table = "/data/ott-qa/new_csv/" + answer_table_uid + ".csv"
                    table = pd.read_csv(table, sep=',', index_col=None)
                    table = table.astype(str)
                    real_answer = eval(data['answer_text_gt'])

                    print('question: ', inp_question)
                    print('Real answer: ', real_answer)
                    #print('usou a tabela gt:', data.top1_flag)
                    print("tapex:")
                    tapex_answer_text, tapex_generated_tokens = question_answering(inp_question, table, model='tapex')      #***
                    print('Resposta: ', tapex_answer_text)
                    # nao correto, verificar
                    #if real_answer.lower().strip() == tapex_answer_text.lower().strip():
                    #    tapex_top1_flag = True
                    #    count_EM_tapex +=1
                    #    print("acertou!")
                    #else:
                    #    tapex_top1_flag = False
                    #    print("errou!")
                    tapex_answer_text_list.append(tapex_answer_text)
                    print("tapas:")
                    #results, aggregation_predictions_string = question_answering(inp_question, table, model='tapas')
                    #result = get_tapas_result(inp_question,results,aggregation_predictions_string)
                    #print('Resposta: ',result[0])
                    #tapas_answer_text_list.append(result[0])
                    #aggregation_predictions_string_list.append(aggregation_predictions_string)
                    #tapas_answer_text_list.append('n/a')
                    #aggregation_predictions_string_list.append('n/a')

                    print("")

                new_data = {'tapex_answer_text'   : tapex_answer_text_list} #,
                            #'tapex_answer_tokens' : tapex_generated_tokens} #,
                            #'tapex_top1_flag'     : tapex_top1_flag},
                            #'tapas_answer_text'   : tapas_answer_text_list,
                            #'tapas_top1_flag'     : tapas_top1_flag,
                            #'tapas_agg_string'    : aggregation_predictions_string_list}

                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                print(index)

                if index % 50 == 0:
                    reader_file = path_to_files['reader_destination'] + retriever_file.split('/')[-1]
                    reader_file = reader_file.replace('csv',f'{index}.csv')
                    
                    df_reader = pd.concat([dataset.reset_index(drop=True), df.reset_index(drop=True)], axis=1)  ## ok
                    df_reader.to_csv(reader_file, sep=',', index=False)       
                    print(f'criado {reader_file}')

                #time.sleep(3)
                #df_reader = pd.concat([dataset.reset_index(drop=True), df.reset_index(drop=True)], axis=1)  ## ok
                #return(df_reader)

            df_reader = pd.concat([dataset.reset_index(drop=True), df.reset_index(drop=True)], axis=1)  ## ok
            return(df_reader)


def remove_questions(file_source, to_remove):
    df = pd.read_csv(file_source, sep=',')
    df_to_remove = pd.read_csv(to_remove, sep=',')
    list_to_remove = df_to_remove['question_id'].to_list()

    print(df.shape)
    df_filtered = df[~df['question_id'].isin(list_to_remove)]
    print(df_filtered.shape)
    new_file = file_source.replace(".csv","_filtered.csv")
    df_filtered.to_csv(new_file, sep=',', index=False)


def concatena_files():
    #file1 = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512.de0a250.csv'
    file1 = '/data/tat-qa/reader_tapex/reranking-cen04/mpnet_table_intro_embeddings_cpu_512_512_filtered.de0a950.csv'
    df_file1 = pd.read_csv(file1, sep=',')
    print(df_file1.shape)
    df_file1.reset_index(drop=True)
    df_file1 = df_file1.drop(df_file1.index[950:])  ## teste
    print(df_file1.shape)

    file2 = '/data/tat-qa/reader_tapex/reranking-cen04/mpnet_table_intro_embeddings_cpu_512_512_filtered.de950.ao.final.csv'

    df_file2 = pd.read_csv(file2, sep=',')
    print(df_file2.shape)


    #file3 = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512de2100a2213.csv'
    #df_file3 = pd.read_csv(file3, sep=',')
    #print(df_file3.shape)

    df_reader = pd.concat([df_file1.reset_index(drop=True), df_file2.reset_index(drop=True)], axis=0)
    print(df_reader.shape)
    reader_file = '/data/tat-qa/reader_tapex/reranking-cen04/mpnet_table_intro_embeddings_cpu_512_512_filtered.csv'
    df_reader.to_csv(reader_file, sep=',', index=False)

def reduz_list_100_10(lst):
    lst = eval(lst)
    return lst[:10]


def concatena_tat_files():

    file1 = '/data/tat-qa/reader_tapex/improved/mpnet_table_embeddings_cpu_512_512_filtered.de0a1000.csv'
    df_file1 = pd.read_csv(file1, sep=',')

    print(df_file1.shape)
    df_file1 = df_file1.drop(df_file1.index[1000:])
    print(df_file1.shape)


    file2 = '/data/tat-qa/reader_tapex/improved/mpnet_table_embeddings_cpu_512_512_filtered.de1000ao.final.csv'
    df_file2 = pd.read_csv(file2, sep=',')

    print(df_file2.shape)

    df = pd.concat([df_file1.reset_index(drop=True), df_file2.reset_index(drop=True)], axis=0)

    print()

    reader_file = '/data/tat-qa/reader_tapex/improved/mpnet_table_embeddings_cpu_512_512_filtered-full.csv'
    df.to_csv(reader_file, sep=',', index=False)
 
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        if pd.isna(text):
            return(text)

        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        if pd.isna(text):
            return(text)

        return " ".join(text.split())

    def remove_punc(text):
        if pd.isna(text):
            return(text)

        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        if pd.isna(text):
            return(text)

        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_list(answer_list):
    normalized_list = []
    for text in answer_list:
        normalized_list.append(normalize_answer(text))
    return normalized_list

def get_tokens(s):
    if pd.isna(s):
            return []
    if not s:
        return []
    return normalize_answer(s).split()

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_f1():
    reader_file = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512_evaluated.csv'
    dataset = pd.read_csv(reader_file, sep=',') 
    
    #dataset = dataset.drop(dataset.index[:2200])  ## teste 

    f1_list = []

    for index, data in dataset.iterrows():
        answer_text_list = eval(data['tapex_answer_text'])
        answer_text_normalized_list = normalize_list(answer_text_list)
        answer_text_gt = normalize_answer(data['answer_text_gt'])
        f1_exact_match = f1_top10 = f1_top50 = f1_top100 = 0.0

        f1_top100_answer_list = []
        gold_answers = []
        gold_answers.append(answer_text_gt)
        prediction = answer_text_normalized_list[0]
        #total = len(answer_text_normalized_list)
        #for prediction in answer_text_normalized_list:
            #f1_top100_answer_list.append(max(compute_f1(a, prediction) for a in gold_answers))
        f1 = max(compute_f1(a, prediction) for a in gold_answers)

        #f1 = sum(f1_top100_answer_list) / total
        f1_list.append(f1)
        print(index)
    

    dataset['reader_f1'] = f1_list

    fout = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512_evaluated.csv'
    dataset.to_csv(fout, sep=',', index=False)
    print()
    print('resumo:')

    print('f1 médio:')
    print(dataset['reader_f1'].sum()/index)




    

def calculate_acuracia():
    reader_file = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512.csv'
    dataset = pd.read_csv(reader_file, sep=',') 

    exact_match_list = []
    top10_list  = []
    top50_list  = []
    top100_list = []

    for index, data in dataset.iterrows():
        answer_text_list = eval(data['tapex_answer_text'])
        answer_text_normalized_list = normalize_list(answer_text_list)
        answer_text_gt = normalize_answer(data['answer_text_gt'])
        exact_match = top10 = top50 = top100 = False
        match = int((answer_text_gt) == (answer_text_normalized_list[0]))
        if match:
            #print("acertou")
            exact_match = True

        elif answer_text_gt in answer_text_normalized_list[0:10]:
            #print("top10")
            top10 = True

        elif answer_text_gt in answer_text_normalized_list[0:50]:
            #print("top50")
            top50 = True   

        elif answer_text_gt in answer_text_normalized_list[0:100]:
            #print("top100")
            top100 = True
        #print(index)
        exact_match_list.append(exact_match)
        top10_list.append(top10)
        top50_list.append(top50)
        top100_list.append(top100)
 
    dataset['reader_exact_match_list'] = exact_match_list
    dataset['reader_top10_list']  = top10_list
    dataset['reader_top50_list']  = top50_list
    dataset['reader_top100_list'] = top100_list
    fout = '/data/ott-qa/reader_tapex/mpnet_table_embeddings_cpu_512_512_evaluated.csv'
    dataset.to_csv(fout, sep=',', index=False)
    print()
    print('resumo acuracia:')
    em = (dataset['reader_exact_match_list'].sum()/index)
    top10 = (dataset['reader_top10_list'].sum()/index)
    top50 = (dataset['reader_top50_list'].sum()/index)
    top100 = (dataset['reader_top100_list'].sum()/index)
    print(em)
    print(em + top10)
    print(em + top10 + top50)
    print(em + top10 + top50 + top100)


def main():

        # parametros de configuracao
        run_improve_question         = False 

        with open('/QA/Bert/code/path_to_files.json', 'r') as file:
            path_to_files = json.load(file)

        retriever_output = [#'mpnet_table_intro_embeddings_cpu_512_512.csv',
                            'mpnet_table_embeddings_cpu_512_512.csv']#,
                            #'mpnet_table_intro_embeddings_cpu_384_512.csv',
                            #'mpnet_table_embeddings_cpu_384_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_512_512.csv',
                            #'mpnet_table_section_title_embeddings_cpu_384_512.csv']
        for retriever_file in retriever_output:
            retriever_file = path_to_files['reader_source'] + retriever_file
            #retriever_file =  f'/data/ott-qa/retriever/improved/{retriever_file}'
            device = 'CPU'
            df_reader = build_reader(retriever_file, "cpu", run_improve_question)
            reader_file = path_to_files['reader_destination'] + retriever_file.split('/')[-1]
            df_reader.to_csv(reader_file, sep=',', index=False)       
            print(f'criado {reader_file}')
            print("")


if __name__ == '__main__':
    print('main')
    #main()
    concatena_files()
    #calculate_acuracia()
        #resumo acuracia:
        #0.027564392227745142
        #0.07094441934026209
        #0.12607320379575238
        #0.16177135110709445
    #calculate_f1()
        #f1 médio:
        # 0.05000436877978676


    #concatena_tat_files()

    #fin = '/data/tat-qa/reader_tapex/filter002/mpnet_table_embeddings_cpu_512_512-full-to-evaluate.csv'
    #to_remove = '/data/tat-qa/to_filter/remover002.unnmamed0.2024.08.10.csv'
    #remove_questions(fin, to_remove)


