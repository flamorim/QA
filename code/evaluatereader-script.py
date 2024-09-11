## FUNCOES FORAM INCORPORADAS NO READER.PY

import json
import re
import collections
import string
import sys
import pandas as pd
import time
import ast
from collections import OrderedDict
import datetime
  
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


def get_tokens(s):
    if pd.isna(s):
            return []
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    match = int(normalize_answer(a_gold) == normalize_answer(a_pred))
    #if match:
    #    print("acertou")
    #else:
    #    print("errou")
    return match


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

def get_raw_scores(gt_list, predictions_list, top_index, fin):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    #exact_scores = {}
    #f1_scores = {}
    
    exact_scores = []
    f1_scores = []
    convert_name = {1:'reader_exact_match_list_filtered',
                    2:'reader_top2_list_filtered',
                    3:'reader_top3_list_filtered',
                    4:'reader_top4_list_filtered',
                    5:'reader_top5_list_filtered',
                    6:'reader_top6_list_filtered',
                    7:'reader_top7_list_filtered',
                    8:'reader_top8_list_filtered',
                    9:'reader_top9_list_filtered',
                    10:'reader_top10_list_filtered',
                    }
    col_name = convert_name[top_index]



    for id,(gt,prediction) in enumerate(zip(gt_list,predictions_list)): # gt
        #qas_id = example['question_id']
        #gold_answers = [reference['reference'][qas_id]]
        #prediction = example['pred']
        #gt = eval(gt)
        #prediction = eval(prediction)
        #prediction = prediction[0]   # pegando apenas a melhor resposta
        #prediction = reference[id]
        #gold_answers = []
        #gold_answers.append(gt)
        #exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        #f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)
        exact_scores.append(max(compute_exact(a, prediction) for a in gt.split()))
        f1_scores.append(max(compute_f1(a, prediction) for a in gt.split()))
        #print(exact_scores[id])
        #print(f1_scores[id])
        #print(gold_answers)
        #print(prediction)
        #print(id)
        #time.sleep(1)
    df = pd.DataFrame()
    df[col_name] = exact_scores
    remove = fin.split('/')[-1]
    fname = fin.replace(remove, col_name + ".csv")
    #fname = f'/data/tat-qa/reader_tapex/improved/{col_name}.csv'
    df.to_csv(fname, sep=',', index=False)
   


    #qid_list = reference['reference'].keys()
    total = len(predictions_list)
    #total = len(examples)

    return collections.OrderedDict(
        [
            ("total exact", 100.0 * sum(exact_scores) / total),
            ("total f1", 100.0 * sum(f1_scores) / total),
            ("total", total),
        ]
    )
    #for k in qid_list:
    #    if k not in exact_scores:
    #        print("WARNING: MISSING QUESTION {}".format(k))
    #qid_list = list(set(qid_list) & set(exact_scores.keys()))

    #return collections.OrderedDict(
    #    [
    #        ("total exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
    #        ("total f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
    #        ("total", total),
    #    ]
    


## reranker - melhor valor - artigo


#reader = False
#reranker = True
#if reader == True:

def evaluate_reader(fin):
    ## reader improved003 
    ## ('total exact', 2.5293586269196027), ('total f1', 4.534104107274842), ('total', 2214)
    ## ref = '/QA/Bert/data/ott-qa/reader_tapex_tapas/improved003/mpnet_table_embeddings_cpu_512_512.csv'

    ## reader improved003 
    ## ([('total exact', 3.3875338753387534), ('total f1', 6.101168140065001), ('total', 2214)])
    #ref = '/QA/Bert/data/ott-qa/reader_tapex_tapas/improved003/mpnet_table_intro_embeddings_cpu_512_512.csv'

    ## reader baseline 
    ## ([('total exact', 3.3875338753387534), ('total f1', 5.8729582171316626), ('total', 2214)])
    #ref = '/QA/Bert/data/ott-qa/reader_tapex_tapas/mpnet_table_intro_embeddings_cpu_512_512.csv'

    ## reader baseline 
    ## [('total exact', 2.7551942186088527), ('total f1', 4.90999490321984), ('total', 2214)])
 
    #ref = '/QA/Bert/data/ott-qa/reranking/improved003/mpnet_table_intro_embeddings_cpu_512_512.csv'

    df_ref = pd.read_csv(fin)
    print(df_ref.columns)
    ground_thruth_list = df_ref.clean_answer_text_gt.to_list()     ## gt
    #prediction_list  = df_ref.tapex_answer_text.to_list()  ## predicao
    print(fin)

    current_datetime = datetime.datetime.now().strftime("%d%m%y %H:%M:%S")
    log_file = fin.replace('.csv','.EM_F1.log')

    with open(log_file, 'w') as file:
        file.write(current_datetime + '\n')

        for count in range(1,11):
            column_name = f'clean_tapex_answer_top{count}_text'
            prediction_list  = df_ref[column_name].to_list()  ## predicao
            print(get_raw_scores(ground_thruth_list, prediction_list, count, fin))
            values_dict = get_raw_scores(ground_thruth_list, prediction_list, count, fin)
            file.write(str(values_dict['total exact']))
            file.write(',')
            file.write(str(values_dict['total f1']))
            file.write(',')
            file.write(str(values_dict['total']))
            file.write('\n')
    
    print('foi 01')


##################


def sum_totals(file_path):
    total_exact_sum = 0
    total_f1_sum = 0
    
    # Abrir o arquivo
    with open(file_path, 'r') as file:
        next(file)  # Pula a primeira linha
        for line in file:
            # Converter a string da linha para um OrderedDict
            line_dict = ast.literal_eval(line.strip())
            
            # Somar os valores de 'total exact' e 'total f1'
            total_exact_sum += line_dict[0]
            total_f1_sum += line_dict[1]
    
    print (total_exact_sum, total_f1_sum)

##################
#if reranker == True:

def evaluate_reranker():
    # verificar
    ref = '/QA/Bert/data/ott-qa/reranking/improved003/mpnet_table_intro_embeddings_cpu_512_512.csv'

    df_ref = pd.read_csv(ref)
    def get_first(x):
        x_list = ast.literal_eval(x)
        return x_list[0]

    #df_ref['rr_top1_table_uid'] = df_ref ['rr_top100_table_uid'].apply(get_first)
    df_ref['ref'] = True


    print(df_ref.columns)
    data = df_ref.ref.to_list()           ## gt
    data = ['acertou' if value else 'errou' for value in data]

    ref  = df_ref.rr_top1_flag.to_list()  ## predicao
    ref = ['acertou' if value else 'errou' for value in ref]
    print(get_raw_scores(data, ref))
    print('foi 01')

#if tapex == True:
#    mpnet_table_intro_embeddings_cpu_512_512

def clean_answer_gt(fin,fout):
    #ref = '/QA/Bert/data/tat-qa/reader_tapex/testes_reader.csv'
    df = pd.read_csv(fin)
    df['clean_answer_text_gt'] = ''
    df['clean_tapex_answer_text'] = ''
    
    def clean_gt(gt):
        old = gt
        gt = ast.literal_eval(gt)
        gt_type = type(gt)
        if gt_type == str:
            #replace
            answer = gt
        elif gt_type == list:
            answer = ''
            for elem in (gt):
                answer = answer + ' ' + elem
                #print(answer)
            answer = answer.strip()
        elif gt_type == float:
            answer = str(gt)
            answer = answer.replace(',','')
        else:
            answer = str(gt)        
        answer = answer.replace('(','').replace(')','').replace('$','').replace('%','').replace(',','')
        print(old, answer)
        return(answer)
            #x_list = ast.literal_eval(x) aqui
        #return x_list[0]

    def clean_tapex_answer(gt,topn):
        old = gt
        gt = ast.literal_eval(gt)[topn]
        gt_type = type(gt)
        if gt_type == str:
            #replace
            answer = gt
        elif gt_type == list:
            answer = ''
            for elem in (gt):
                answer = answer + ' ' + elem
                #print(answer)
            answer = answer.strip()
        elif gt_type == float:
            answer = str(gt)
            answer = answer.replace(',','')
        else:
            answer = str(gt)        
        answer = answer.replace('(','').replace(')','').replace('$','').replace('%','').replace(',','')
        print(old, answer)
        return(answer)
            #x_list = ast.literal_eval(x) aqui
        #return x_list[0]





    df['clean_answer_text_gt']    = df['answer_text_gt'].apply(clean_gt)
    df['clean_tapex_answer_top1_text'] = df['tapex_answer_text'].apply(clean_tapex_answer,topn=0)
    df['clean_tapex_answer_top2_text'] = df['tapex_answer_text'].apply(clean_tapex_answer,topn=1)
    df['clean_tapex_answer_top3_text'] = df['tapex_answer_text'].apply(clean_tapex_answer,topn=2)
    df['clean_tapex_answer_top4_text'] = df['tapex_answer_text'].apply(clean_tapex_answer,topn=3)
    df['clean_tapex_answer_top5_text'] = df['tapex_answer_text'].apply(clean_tapex_answer,topn=4)
    df['clean_tapex_answer_top6_text'] = df['tapex_answer_text'].apply(clean_tapex_answer,topn=5)
    df['clean_tapex_answer_top7_text'] = df['tapex_answer_text'].apply(clean_tapex_answer,topn=6)
    df['clean_tapex_answer_top8_text'] = df['tapex_answer_text'].apply(clean_tapex_answer,topn=7)
    df['clean_tapex_answer_top9_text'] = df['tapex_answer_text'].apply(clean_tapex_answer,topn=8)
    df['clean_tapex_answer_top10_text'] = df['tapex_answer_text'].apply(clean_tapex_answer,topn=9)
    
    df.to_csv(fout, sep=',', index=False)





def main():


    #fin = '/data/tat-qa/reader_tapex/improved_intro/mpnet_table_intro_embeddings_cpu_512_512_filtered.csv'
    #fout = '/data/tat-qa/reader_tapex/improved_intro/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'  #verificar

    #fin = '/data/tat-qa/reader_table_opt/mpnet_table_embeddings_cpu_512_512.filtered.csv'
    #fout = '/data/tat-qa/reader_table_opt/mpnet_table_embeddings_cpu_512_512.filtered-to-evaluate.csv'  #verificar

    #fin = '/data/tat-qa/reader_table_opt/newimproved_e_filter002/mpnet_table_intro_embeddings_cpu_512_512_filtered.csv'
    #fout = '/data/tat-qa/reader_table_opt/newimproved_e_filter002/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'

    #fin  = '/data/tat-qa/reader_tapex/improved/mpnet_table_embeddings_cpu_512_512_filtered-full.csv'
    #fout = '/data/tat-qa/reader_tapex/improved/mpnet_table_embeddings_cpu_512_512_filtered-to-evaluate.csv'

    #fin  = '/data/tat-qa/reader_tapex/reranking-cen04/mpnet_table_intro_embeddings_cpu_512_512_filtered.csv'
    #fout = '/data/tat-qa/reader_tapex/reranking-cen04/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'

    #fin  = '/data/tat-qa/reader_table_opt/reranking-table-opt/mpnet_table_intro_embeddings_cpu_512_512_filtered.csv'
    #fout = '/data/tat-qa/reader_table_opt/reranking-table-opt/mpnet_table_intro_embeddings_cpu_512_512_filtered.csv-to-evaluate.csv'

    #fin = '/data/tat-qa/reader_table_opt/reranking-table-opt/refazendosemtableopt/mpnet_table_intro_embeddings_cpu_512_512_filtered.csv'
    #fout = '/data/tat-qa/reader_table_opt/reranking-table-opt/refazendosemtableopt/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'
    
    #fin = '/data/tat-qa/reader_table_opt/reranking-table-opt-new-score5/mpnet_table_intro_embeddings_cpu_512_512_filtered.csv'
    #fout = '/data/tat-qa/reader_table_opt/reranking-table-opt-new-score5/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'

    #fin = '/data/tat-qa/reader_table_opt/reranking-table-opt-new-score8/mpnet_table_intro_embeddings_cpu_512_512_filtered.csv'
    #fout = '/data/tat-qa/reader_table_opt/reranking-table-opt-new-score8/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'

    ### NOVO PROMPT

    fin = '/data/tat-qa/reader_table_opt/new-rr-cenario4-score7/mpnet_table_intro_embeddings_cpu_512_512_filtered.csv'
    fout ='/data/tat-qa/reader_table_opt/new-rr-cenario4-score7/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'

    clean_answer_gt(fin,fout)
    print('fim')


    #fin = '/data/tat-qa/reader_tapex/improved_intro/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'  #verificar
    #fin = '/data/tat-qa/reader_table_opt/newimproved_e_filter002/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'
    #fin = '/data/tat-qa/reader_table_opt/filter002/mpnet_table_embeddings_cpu_512_512-full-to-evaluate_filtered.csv'
    #fin = '/data/tat-qa/reader_tapex/reranking-cen04/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'
    #fin = '/data/tat-qa/reader_table_opt/reranking-table-opt/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'
    #fin = '/data/tat-qa/reader_table_opt/reranking-table-opt-new-score2/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'
    #fin = '/data/tat-qa/reader_table_opt/reranking-table-opt-new-score5/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'
    fin ='/data/tat-qa/reader_table_opt/new-rr-cenario4-score7/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'

    evaluate_reader(fin) 
    print('fim')


    fin ='/data/tat-qa/reader_table_opt/new-rr-cenario4-score7/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.EM_F1.log'
    sum_totals(fin)



if __name__ == '__main__':

    main()
    

#TAT-QA com as perguntas otimizadas ** ERROR
#OrderedDict([('total exact', 0.9569377990430622), ('total f1', 1.0455431508063087), ('total', 1881)])
#OrderedDict([('total exact', 0.4784688995215311), ('total f1', 0.5404926457558037), ('total', 1881)])
#OrderedDict([('total exact', 0.3189792663476874), ('total f1', 0.35442140705298597), ('total', 1881)])
#OrderedDict([('total exact', 0.4253056884635832), ('total f1', 0.4607478291688818), ('total', 1881)])
#OrderedDict([('total exact', 0.4784688995215311), ('total f1', 0.523577078601002), ('total', 1881)])
#OrderedDict([('total exact', 0.4253056884635832), ('total f1', 0.48834206728943574), ('total', 1881)])
#OrderedDict([('total exact', 0.3189792663476874), ('total f1', 0.41318074559071793), ('total', 1881)])
#OrderedDict([('total exact', 0.4253056884635832), ('total f1', 0.4784688995215311), ('total', 1881)])
#OrderedDict([('total exact', 0.2658160552897395), ('total f1', 0.27910685805422647), ('total', 1881)])
#OrderedDict([('total exact', 0.3721424774056353), ('total f1', 0.3721424774056353), ('total', 1881)])

#TAT-QA baseline
#OrderedDict([('total exact', 7.9213184476342375), ('total f1', 8.875022327503796), ('total', 1881)])
#OrderedDict([('total exact', 3.2429558745348217), ('total f1', 4.408182976294432), ('total', 1881)])
#OrderedDict([('total exact', 3.5619351408825093), ('total f1', 4.398546844207488), ('total', 1881)])
#OrderedDict([('total exact', 2.977139819245082), ('total f1', 3.663071820966558), ('total', 1881)])
#OrderedDict([('total exact', 2.977139819245082), ('total f1', 3.6821117200935043), ('total', 1881)])
#OrderedDict([('total exact', 3.4024455077086655), ('total f1', 3.9954031039302356), ('total', 1881)])
#OrderedDict([('total exact', 3.0303030303030303), ('total f1', 3.607179729063385), ('total', 1881)])
#OrderedDict([('total exact', 2.8708133971291865), ('total f1', 3.732713326014761), ('total', 1881)])
#OrderedDict([('total exact', 3.508771929824561), ('total f1', 4.038108356290175), ('total', 1881)])
#OrderedDict([('total exact', 3.0834662413609784), ('total f1', 3.8395445653363502), ('total', 1881)])

# com as tabelas otimizadas
#run_reader_improve_tables.py
#OrderedDict([('total exact', 6.432748538011696), ('total f1', 7.574534316489474), ('total', 1881)])
#OrderedDict([('total exact', 3.721424774056353), ('total f1', 4.596859905346434), ('total', 1881)])
#OrderedDict([('total exact', 3.880914407230197), ('total f1', 4.652709214112722), ('total', 1881)])
#OrderedDict([('total exact', 3.136629452418926), ('total f1', 3.820958119203733), ('total', 1881)])
#OrderedDict([('total exact', 3.189792663476874), ('total f1', 3.8953444216602118), ('total', 1881)])
#OrderedDict([('total exact', 3.0834662413609784), ('total f1', 3.782122191733176), ('total', 1881)])
#OrderedDict([('total exact', 3.0303030303030303), ('total f1', 3.662020831079842), ('total', 1881)])
#OrderedDict([('total exact', 3.29611908559277), ('total f1', 3.888611978723621), ('total', 1881)])
#OrderedDict([('total exact', 3.0303030303030303), ('total f1', 3.706277478207302), ('total', 1881)])
#OrderedDict([('total exact', 3.668261562998405), ('total f1', 4.221490010963695), ('total', 1881)])

# com as tabelas otimizadas pela segunda vez e a introducao acrescentada no reader
#run_reader_improve_tables.py
#OrderedDict([('total exact', 6.3264221158958005), ('total f1', 7.340567165128569), ('total', 1881)])
#OrderedDict([('total exact', 3.136629452418926), ('total f1', 4.094668296751044), ('total', 1881)])
#OrderedDict([('total exact', 3.508771929824561), ('total f1', 4.291160081493759), ('total', 1881)])
#OrderedDict([('total exact', 3.6150983519404574), ('total f1', 4.405774537353485), ('total', 1881)])
#OrderedDict([('total exact', 3.136629452418926), ('total f1', 3.7120141822128714), ('total', 1881)])
#OrderedDict([('total exact', 3.4556087187666136), ('total f1', 3.854704121900244), ('total', 1881)])
#OrderedDict([('total exact', 2.9239766081871346), ('total f1', 3.533457600963321), ('total', 1881)])
#OrderedDict([('total exact', 3.2429558745348217), ('total f1', 4.004305252311632), ('total', 1881)])
#OrderedDict([('total exact', 3.0834662413609784), ('total f1', 3.7622929593136445), ('total', 1881)])
#OrderedDict([('total exact', 3.189792663476874), ('total f1', 3.6778840853543744), ('total', 1881)])

# apenas com a introducao acrescentada no reader (
#entrada: tat-qa\reader_tapex\improved_intro)
    #fin = '/data/tat-qa/reader_tapex/improved_intro/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv', que foi gerado a partir:
    #"reader_source"        : "/data/tat-qa/retriever/improved/improved_e_filter002/",
#saida:
#OrderedDict([('total exact', 7.442849548112706), ('total f1', 8.458577174216538), ('total', 1881)])
#OrderedDict([('total exact', 4.30622009569378),  ('total f1', 5.361714309082731), ('total', 1881)])
#OrderedDict([('total exact', 3.5619351408825093),('total f1', 4.251944449927076), ('total', 1881)])
#OrderedDict([('total exact', 2.5518341307814993),('total f1', 3.3851168324852523), ('total', 1881)])
#OrderedDict([('total exact', 3.29611908559277),  ('total f1', 3.910962309733839), ('total', 1881)])
#OrderedDict([('total exact', 2.9239766081871346),('total f1', 3.4622206816811865), ('total', 1881)])
#OrderedDict([('total exact', 2.977139819245082), ('total f1', 3.601767274016077), ('total', 1881)])
#OrderedDict([('total exact', 3.4556087187666136),('total f1', 4.128785444574919), ('total', 1881)])
#OrderedDict([('total exact', 2.7113237639553427),('total f1', 3.485915968372108), ('total', 1881)])
#OrderedDict([('total exact', 3.189792663476874), ('total f1', 3.6994389625968576), ('total', 1881)])


#usando o reader com a saída do reranking cenario 4 ( o melhor de todos)
#mpnet_table_intro_embeddings_cpu_512_512_filtered.csv'
#"reader_source"        : "/data/tat-qa/reranking/intro_improved_filter/",
#"reader_destination"   : "/data/tat-qa/reader_tapex/reranking-cen04/",
#   COM IMPROVED QUESTION :)ERRO
#OrderedDict([('total exact', 0.9569377990430622), ('total f1', 1.0455431508063087), ('total', 1881)])
#OrderedDict([('total exact', 0.4784688995215311), ('total f1', 0.5404926457558037), ('total', 1881)])
#OrderedDict([('total exact', 0.3189792663476874), ('total f1', 0.35442140705298597), ('total', 1881)])
#OrderedDict([('total exact', 0.4253056884635832), ('total f1', 0.4607478291688818), ('total', 1881)])
#OrderedDict([('total exact', 0.4784688995215311), ('total f1', 0.523577078601002), ('total', 1881)])
#OrderedDict([('total exact', 0.4253056884635832), ('total f1', 0.48834206728943574), ('total', 1881)])
#OrderedDict([('total exact', 0.3189792663476874), ('total f1', 0.41318074559071793), ('total', 1881)])
#OrderedDict([('total exact', 0.4253056884635832), ('total f1', 0.4784688995215311), ('total', 1881)])
#OrderedDict([('total exact', 0.2658160552897395), ('total f1', 0.27910685805422647), ('total', 1881)])
#OrderedDict([('total exact', 0.3721424774056353), ('total f1', 0.3721424774056353), ('total', 1881)])

# COM otimizacao, eliminando somente score = 0
#/data/tat-qa/reader_table_opt/reranking-table-opt-new/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv
#OrderedDict([('total exact', 10.738968633705475), ('total f1', 12.317692392980877), ('total', 1881)])
#OrderedDict([('total exact', 4.891015417331206), ('total f1', 6.123064416576332), ('total', 1881)])
#OrderedDict([('total exact', 4.359383306751727), ('total f1', 5.169789275052434), ('total', 1881)])
#OrderedDict([('total exact', 3.6150983519404574), ('total f1', 4.221857060772533), ('total', 1881)])
#OrderedDict([('total exact', 3.29611908559277), ('total f1', 3.876694575699141), ('total', 1881)])
#OrderedDict([('total exact', 2.7644869750132908), ('total f1', 3.4286447205107495), ('total', 1881)])
#OrderedDict([('total exact', 2.9239766081871346), ('total f1', 3.7159753826420494), ('total', 1881)])
#OrderedDict([('total exact', 3.189792663476874), ('total f1', 3.652095403297564), ('total', 1881)])
#OrderedDict([('total exact', 2.977139819245082), ('total f1', 3.560945522668011), ('total', 1881)])
#OrderedDict([('total exact', 2.7113237639553427), ('total f1', 3.2092577121816883), ('total', 1881)])

# COM otimizacao, eliminando somente score <= 0.2
#ficou exatamente igual sem otimizacao, isto é, não piorou o que é bom
#/data/tat-qa/reader_table_opt/reranking-table-opt-new-score2/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv
#OrderedDict([('total exact', 10.738968633705475), ('total f1', 12.317692392980877), ('total', 1881)])
#OrderedDict([('total exact', 4.891015417331206), ('total f1', 6.123064416576332), ('total', 1881)])
#OrderedDict([('total exact', 4.359383306751727), ('total f1', 5.169789275052434), ('total', 1881)])
#OrderedDict([('total exact', 3.6150983519404574), ('total f1', 4.221857060772533), ('total', 1881)])
#OrderedDict([('total exact', 3.29611908559277), ('total f1', 3.876694575699141), ('total', 1881)])
#OrderedDict([('total exact', 2.7644869750132908), ('total f1', 3.4286447205107495), ('total', 1881)])
#OrderedDict([('total exact', 2.9239766081871346), ('total f1', 3.7159753826420494), ('total', 1881)])
#OrderedDict([('total exact', 3.189792663476874), ('total f1', 3.652095403297564), ('total', 1881)])
#OrderedDict([('total exact', 2.977139819245082), ('total f1', 3.560945522668011), ('total', 1881)])
#OrderedDict([('total exact', 2.7113237639553427), ('total f1', 3.2092577121816883), ('total', 1881)])

# COM otimizacao, eliminando somente score <= 0.5
#/data/tat-qa/reader_table_opt/reranking-table-opt-new-score5/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv
#OrderedDict([('total exact', 10.738968633705475), ('total f1', 12.317692392980877), ('total', 1881)])
#OrderedDict([('total exact', 4.891015417331206), ('total f1', 6.123064416576332), ('total', 1881)])
#OrderedDict([('total exact', 4.359383306751727), ('total f1', 5.169789275052434), ('total', 1881)])
#OrderedDict([('total exact', 3.6150983519404574), ('total f1', 4.221857060772533), ('total', 1881)])
#OrderedDict([('total exact', 3.29611908559277), ('total f1', 3.876694575699141), ('total', 1881)])
#OrderedDict([('total exact', 2.7644869750132908), ('total f1', 3.4286447205107495), ('total', 1881)])
#OrderedDict([('total exact', 2.9239766081871346), ('total f1', 3.7159753826420494), ('total', 1881)])
#OrderedDict([('total exact', 3.189792663476874), ('total f1', 3.652095403297564), ('total', 1881)])
#OrderedDict([('total exact', 2.977139819245082), ('total f1', 3.560945522668011), ('total', 1881)])
#OrderedDict([('total exact', 2.7113237639553427), ('total f1', 3.2092577121816883), ('total', 1881)])

# COM otimizacao, eliminando somente score <= 0.8
#/data/tat-qa/reader_table_opt/reranking-table-opt-new-score5/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv
#OrderedDict([('total exact', 10.738968633705475), ('total f1', 12.317692392980877), ('total', 1881)])
#OrderedDict([('total exact', 4.891015417331206), ('total f1', 6.123064416576332), ('total', 1881)])
#OrderedDict([('total exact', 4.359383306751727), ('total f1', 5.169789275052434), ('total', 1881)])
#OrderedDict([('total exact', 3.6150983519404574), ('total f1', 4.221857060772533), ('total', 1881)])
#OrderedDict([('total exact', 3.29611908559277), ('total f1', 3.876694575699141), ('total', 1881)])
#OrderedDict([('total exact', 2.7644869750132908), ('total f1', 3.4286447205107495), ('total', 1881)])
#OrderedDict([('total exact', 2.9239766081871346), ('total f1', 3.7159753826420494), ('total', 1881)])
#OrderedDict([('total exact', 3.189792663476874), ('total f1', 3.652095403297564), ('total', 1881)])
#OrderedDict([('total exact', 2.977139819245082), ('total f1', 3.560945522668011), ('total', 1881)])
#OrderedDict([('total exact', 2.7113237639553427), ('total f1', 3.2092577121816883), ('total', 1881)])


# COM otimizacao, novo prompt

#/data/tat-qa/reader_table_opt/new-rr-cenario4-score0/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv
#OrderedDict([('total exact', 10.579479000531633), ('total f1', 12.173112314190268), ('total', 1881)])
#OrderedDict([('total exact', 4.9973418394471025), ('total f1', 6.196412488931356), ('total', 1881)])
#OrderedDict([('total exact', 4.30622009569378), ('total f1', 5.12033126068214), ('total', 1881)])
#OrderedDict([('total exact', 3.6150983519404574), ('total f1', 4.138614058869244), ('total', 1881)])
#OrderedDict([('total exact', 2.977139819245082), ('total f1', 3.592640806796886), ('total', 1881)])
#OrderedDict([('total exact', 2.977139819245082), ('total f1', 3.6232043765583493), ('total', 1881)])
#OrderedDict([('total exact', 3.29611908559277), ('total f1', 4.146540593909015), ('total', 1881)])
#OrderedDict([('total exact', 2.977139819245082), ('total f1', 3.4346642975773936), ('total', 1881)])
#OrderedDict([('total exact', 2.7644869750132908), ('total f1', 3.3455072078627333), ('total', 1881)])
#OrderedDict([('total exact', 2.3391812865497075), ('total f1', 2.7625601602209784), ('total', 1881)])

#/data/tat-qa/reader_table_opt/new-rr-cenario4-score1/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv
#OrderedDict([('total exact', 10.260499734183945), ('total f1', 11.92856154332371), ('total', 1881)])
#OrderedDict([('total exact', 5.05050505050505), ('total f1', 6.178691418578706), ('total', 1881)])
#OrderedDict([('total exact', 4.093567251461988), ('total f1', 4.907678416450349), ('total', 1881)])
#OrderedDict([('total exact', 3.508771929824561), ('total f1', 4.036844483415458), ('total', 1881)])
#OrderedDict([('total exact', 2.977139819245082), ('total f1', 3.6133153888749785), ('total', 1881)])
#OrderedDict([('total exact', 2.7113237639553427), ('total f1', 3.378653605691789), ('total', 1881)])
#OrderedDict([('total exact', 3.349282296650718), ('total f1', 4.199703804966963), ('total', 1881)])
#OrderedDict([('total exact', 2.8708133971291865), ('total f1', 3.3637800161667966), ('total', 1881)])
#OrderedDict([('total exact', 2.7113237639553427), ('total f1', 3.273610293860556), ('total', 1881)])
#OrderedDict([('total exact', 2.2328548644338118), ('total f1', 2.6562337381050827), ('total', 1881)])


#/data/tat-qa/reader_table_opt/new-rr-cenario4-score2/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv
#OrderedDict([('total exact', 10.260499734183945), ('total f1', 11.92856154332371), ('total', 1881)])
#OrderedDict([('total exact', 4.9973418394471025), ('total f1', 6.118439779379698), ('total', 1881)])
#OrderedDict([('total exact', 4.040404040404041), ('total f1', 4.860274911983132), ('total', 1881)])
#OrderedDict([('total exact', 3.29611908559277), ('total f1', 3.8206474251131364), ('total', 1881)])
#OrderedDict([('total exact', 2.9239766081871346), ('total f1', 3.571544294472304), ('total', 1881)])
#OrderedDict([('total exact', 2.658160552897395), ('total f1', 3.3609325353391393), ('total', 1881)])
#OrderedDict([('total exact', 3.349282296650718), ('total f1', 4.159831396673502), ('total', 1881)])
#OrderedDict([('total exact', 2.604997341839447), ('total f1', 3.097963960877057), ('total', 1881)])
#OrderedDict([('total exact', 2.7113237639553427), ('total f1', 3.2850024105158306), ('total', 1881)])
#OrderedDict([('total exact', 2.1796916533758637), ('total f1', 2.6030705270471346), ('total', 1881)])

#/data/tat-qa/reader_table_opt/new-rr-cenario4-score5/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv
#OrderedDict([('total exact', 10.1010101010101), ('total f1', 11.741991290963986), ('total', 1881)])
#OrderedDict([('total exact', 4.7315257841573635), ('total f1', 5.885469003542323), ('total', 1881)])
#OrderedDict([('total exact', 3.880914407230197), ('total f1', 4.721193179334147), ('total', 1881)])
#OrderedDict([('total exact', 3.508771929824561), ('total f1', 4.072413203194703), ('total', 1881)])
#OrderedDict([('total exact', 3.136629452418926), ('total f1', 3.632217863917804), ('total', 1881)])
#OrderedDict([('total exact', 2.977139819245082), ('total f1', 3.628014381368353), ('total', 1881)])
#OrderedDict([('total exact', 3.4556087187666136), ('total f1', 4.226285410495937), ('total', 1881)])
#OrderedDict([('total exact', 2.817650186071239), ('total f1', 3.3210741839872804), ('total', 1881)])
#OrderedDict([('total exact', 2.8708133971291865), ('total f1', 3.4558658155516224), ('total', 1881)])
#OrderedDict([('total exact', 2.3391812865497075), ('total f1', 2.7403947579386165), ('total', 1881)])



