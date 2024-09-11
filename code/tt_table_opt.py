import pandas as pd
# resposta reader com opt
answer_opt = '/QA/Bert/data/tat-qa/reader_table_opt/reranking-table-opt/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'

# resposta reader padrao sem opt
answer_std = '/QA/Bert/data/tat-qa/reader_table_opt/reranking-table-opt/refazendosemtableopt/mpnet_table_intro_embeddings_cpu_512_512_filtered-to-evaluate.csv'

# tabela llm opt
table_scores = '/data/tat-qa/llm_table_opt/fromreranking/mpnet_table_intro_embeddings_cpu_512_512_filtered.full.csv'


df_opt = pd.read_csv(answer_opt,      sep=',') 
df_std = pd.read_csv(answer_std,      sep=',') 
df_scores = pd.read_csv(table_scores, sep=',')
df_scores = df_scores[df_scores.position == 'top1']

print(df_opt.shape)
print(df_std.shape)
print(df_scores.shape)

df = pd.DataFrame()

df.insert(0, 'question_idx', df_std['question_idx'])
df.insert(1, 'question_id', df_std['question_id'])
df.insert(2, 'question_txt', df_std['question_txt'])
df.insert(3, 'answer_text_gt', df_std['answer_text_gt'])
df.insert(4, 'std_answer', df_std['clean_tapex_answer_top1_text'])
df.insert(5, 'opt_answer', df_opt['clean_tapex_answer_top1_text'])
df.insert(6, 'std_rr_top1_flag',df_std['rr_top1_flag'])
df.insert(7, 'opt_rr_top1_flag',df_opt['rr_top1_flag'])




print()


Index(['question_idx', 'question_id', 'question_txt', 'table_uid_gt',
       'answer_text_gt', 'answer_type', 'answer_from', 'top100_table_idx',
       'top100_table_score', 'table_idx_gt', 'table_url_gt',
       'table_token_len_gt', 'table_header_gt', 'top100_table_token_len',
       'top100_table_uid', 'top1_flag', 'top10_flag', 'top50_flag',
       'top100_flag', 'top100_table_intro', 'rr_top100_table_idx',
       'rr_top100_table_uid', 'rr_top100_table_score', 'rr_top1_flag',
       'rr_top10_flag', 'rr_top50_flag', 'rr_top100_flag', 'tapex_answer_text',
       'tapas_answer_text', 'tapas_agg_string', 'status_table_opt',
       'clean_answer_text_gt', 'clean_tapex_answer_text',
       'clean_tapex_answer_top1_text', 'clean_tapex_answer_top2_text',
       'clean_tapex_answer_top3_text', 'clean_tapex_answer_top4_text',
       'clean_tapex_answer_top5_text', 'clean_tapex_answer_top6_text',
       'clean_tapex_answer_top7_text', 'clean_tapex_answer_top8_text',
       'clean_tapex_answer_top9_text', 'clean_tapex_answer_top10_text'],
      dtype='object')