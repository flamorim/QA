
dataset com 8891 tabelas em csv:
/QA/Bert/data/ott-qa/csv

dataset com 2214 qa:
/QA/Bert/data/ott-qa/released_data/dev.json 

-------------------------------------------

1) download_dataset_ott-qa-embedding.py
/QA/Bert/Langchain/download_dataset_ott-qa_embeddings.py

1.a) faz o download do dataset e transforma num pkl
entrada: '/data/ott-qa/traindev_tables.json'
saída: /data/ott-qa/embeddings/new_dataset.pkl

1.b) lê o pkl do dataset e cria os embeddings.
entrada: /data/ott-qa/embeddings/new_dataset.pkl
saída: /data/ott-qa/embeddings/mpnet_table_intro_embeddings_{device}_{max_seq}_{max_pos}.pkl

Usa srm sem gtp e com proxy para baixar o modelo para um diretorio local
faz o download e cria os embedding seguindo os modelos

1.c) Cria um csv para dada tabela do dataset
entrada: traindev_tables.json
saída: /data/ott-qa/new_csv/*.csv
    
    create_dataset()
    read_dataset_pickle()
    create_embeddings_table("cpu")
    create_embeddings_table_intro("cpu")
    create_embeddings_table_header("cpu")
    create_embeddings_table_section_text("cpu")
    create_embeddings_table_section_title("cpu")
    create_csv()
    download_released_data_dir()



2) retriever.py
/QA/Bert/Langchain/retriever.py
entradas:
    os embeddings /data/ott-qa/embeddings/mpnet_table_intro_embeddings_{device}_{max_seq}_{max_pos}.pkl
    as perguntas e respostas: /data/ott-qa/released_data/dev.json'
saída: /data/ott-qa/output/mpnet_table_intro_embeddings_{device}_{max_seq}_{max_pos}.json

roda o retriever e gera um json em /data/ott-qa/output/ por cenário com uma linha para cada question



3) calcula_metrica_retriever
confere performance

4) re-ranking
/QA/Bert/bert_q-wqt.py
pega a saída do retriever *.json e faz o novo top list com biencoder
saída: mpnet_RR_**_cpu_384_514.json

5) calcula_metrica_rerank
confere performance

a) modelo antigo
b) novo modelo

6) reranking com LLM (rr-LLM)
re-rank-llm.py

reduzir as tabelas da top-list
para cada qa, pegar a top list e reduzir o tamanho da tabela
entrada: mpnet_**_cpu_384_514.json (saída do retriever, MAS FAZENDO NA SAIDA DO RETRIEVER)
saída mpnet_RR_LLM**_cpu_384_514.json ou mpnet_RR_LLM**_cpu_384_514.json
==> feito para 25 perguntas

7) código para comparar o tamanho das tabelas depois de reduzidas
calcula_metrica_rerank_llm.py





========================================================================================
Who created the series in which the character of Robert , played by actor Nonso Anozie , appeared ?
{
  "question": "Who created the series in which the character of Robert, played by actor Nonso Anozie, appeared?",
  "column_name": ["Year", "Character", "Actor (s)", "Series"],
  "score": [0.2, 0.8, 0.6, 1]
}
Aguardando por 8 segundos...
Who created the series in which the character of Robert , played by actor Nonso Anozie , appeared ?
{
  "question": "Who created the series in which the character of Robert, played by actor Nonso Anozie, appeared?",
  "column_name": ["Year", "Title", "Role"],
  "score": [0.5, 0.5, 0.0]

Who created the series in which the character of Robert , played by actor Nonso Anozie , appeared ?
{
  "question": "Who created the series in which the character of Robert, played by actor Nonso Anozie, appeared?",
  "column_name": ["Year", "Title", "Role", "Notes"],
  "score": [0.2, 0.4, 0.8, 0.6]


{
  "question": "Who created the series in which the character of Robert, played by actor Nonso Anozie, appeared?",
  "column_name": ["Year (s)", "Title", "Role", "Director (s)", "Performance history"],
  "score": [0.2, 0.4, 0.8, 0.6, 0.2]
}
table id Robert_Bathurst_filmography_2


{
  "question": "Who created the series in which the character of Robert, played by actor Nonso Anozie, appeared?",
  "column_name": ["Year", "Title", "Role", "Notes"],
  "score": [0.2, 0.4, 0.8, 0.6]
}
table id Anthony_Head_3

{
  "question": "What did the 2nd championship win at the Sevens Grand Prix Series for the team with the most top 4 finishes qualify them for?",
  "column_name": ["Pos", "No", "Driver", "Constructor", "Time", "Gap"],
  "score": [0.5, 0.5, 1, 1, 0.5, 0.5]
}
table id 2004_French_Grand_Prix_0


{
  "question": "This 70's Kishore Kumar song was in a film produced by Alankar Chitra and directed by Shanker Mukherjee?",
  "column_name": ["Year", "Song", "Film", "Music Director", "Lyricist", "Singer (s)"],
  "score": [0.5, 1, 1, 0.2, 0.2, 0.2]
}
table id Binaca_Geetmala_annual_list_1978_0

What is the title for the Taiwanese television series where Jin Chao-chun plays a Chinese politician who was born in the year 1090 ?
{
  "question": "What is the title for the Taiwanese television series where Jin Chao-chun plays a Chinese politician who was born in the year 1090 ?",
  "column_name": ["Title", "Director", "Cast", "Genre"],
  "score": [1, 0, 0, 0]
}
table id List_of_Hong_Kong_films_of_2002_0


  "question": "What is the full name of the earliest player of the month of the year ?",
  "column_name": ["Month", "Year", "Nationality", "Player", "Team", "Position"],
  "score": [0.9, 0.9, 0.2, 1, 0.8, 0.5]
}
table id SJPF_Segunda_Liga_Player_of_the_Month_0



na OFC_Champions_League_1.csv e Heartland_Championship_1.csv e European_Challenge_Cup_1.csv
output_dict['column_name']
['page_content', 'metadata']


This 70 's Kishore Kumar song was in a film produced by Alankar Chitra and directed by Shanker Mukherjee ?
{
  "question": "This 70's Kishore Kumar song was in a film produced by Alankar Chitra and directed by Shanker Mukherjee?",
  "column_name": ["Year","Film","Song ( s )","Music director ( s )",
    "Language","Co-singer ( s )"
  ],
  "score": [1, 1, 1, 0.5, 0, 0.5]
}
table id Bela_Shende_0



Exception has occurred: InvalidRequestError       (note: full exception trace is shown but execution is paused at: <module>)
The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766
  File "/QA/Bert/code/re-rank-llm.py", line 282, in main
    resposta = chain.invoke(input_data_dict)
  File "/QA/Bert/code/re-rank-llm.py", line 431, in <module> (Current frame)
    main()

    Para obter mais informações sobre como os dados são processados para filtragem de conteúdo e monitoramento de abuso
    Neural de várias classes destinados a detectar e filtrar conteúdos perigosos. Os modelos abrangem quatro categorias (ódio, sexual, violência e automutilação) em quatro níveis de severidade (seguro, baixo, médio e alto). O conteúdo detectado no nível de gravidade "seguro" é rotulado em anotações, mas não está sujeito a filtragem e não é configurável.
Outros modelos de classificação opcionais destinados a detectar risco de jailbreak e conteúdo conhecido para texto e código; esses modelos são classificadores binários que sinalizam se o comportamento do usuário ou do modelo se qualifica como um ataque de jailbreak ou corresponde a um texto conhecido ou código-fonte. O uso desses modelos é opcional, mas o uso do modelo de código de material protegido pode ser necessário para a cobertura do Compromisso de Direitos Autorais do Cliente.

What language is the film based on the J. M. Coetzee novel In the Heart of the Country in ?
{
  "question": "What language is the film based on the J. M. Coetzee novel In the Heart of the Country in ?",
  "column_name": [
    "Title",
    "Director",
    "Country",
    "Genre",
    "Cast",
    "Notes"
  ],
  "score": [
    0.5,
    0.5,
    1,
    0.5,
    0.5,
    0.5
  ]
}
table id List_of_lesbian._gay._bisexual_or_transgender-related_films_of_1981_0
57
------------------------------------------------------------------------------