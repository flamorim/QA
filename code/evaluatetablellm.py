import pandas as pd
from collections import Counter
 
#fin = '/data/tat-qa/llm_table_opt/fromreranking/mpnet_table_intro_embeddings_cpu_512_512_filtered.full.csv'
fin = '/data/tat-qa/llm_table_opt/fromrerankingcenario4/mpnet_table_intro_embeddings_cpu_512_512_filtered.full.csv'


df = pd.read_csv(fin, sep=',')
df = df[df['position'] == 'top1']
antes = df.shape[0]
df = df[df['status'] == 'OK']
depois = df.shape[0]
errors = antes - depois


# Inicializa um Counter para contar as ocorrências
contador = Counter()

# Percorre todas as linhas do DataFrame e atualiza o contador

for count,lista in enumerate(df['column_scores']):
    contador.update(eval(lista))
    print(lista)
    print(df.iloc[count,5])

# Exibe o resultado
print(contador)

# Convertendo o contador para um DataFrame, se necessário
df_contador = pd.DataFrame(list(contador.items()), columns=['Número', 'Frequência'])


df_contador = df_contador.sort_values(by='Número', ascending=False)
nova_linha = {'Número': 'error', 'Frequência': errors}
df_contador.loc[len(df_contador)] = nova_linha
df_contador['percentual'] = round(df_contador['Frequência'] / df_contador['Frequência'].sum() * 100,2)

print(df_contador)


#fin = '/data/tat-qa/llm_table_opt/fromreranking/mpnet_table_intro_embeddings_cpu_512_512_filtered.full.csv'
# Número  Frequência  percentual
#13   61.9           1        0.02
#11    8.3           1        0.02
#14    7.4           1        0.02
#17    3.7           1        0.02
#16    3.6           1        0.02
#25    3.2           2        0.03
#24    2.5           2        0.03
#18    2.3           1        0.02
#3     1.0        1142       18.71
#5     0.9         352        5.77
#2     0.8         797       13.06
#9     0.7         178        2.92
#7     0.6         223        3.65
#0     0.5        1080       17.69
#8     0.4         169        2.77
#10    0.3          70        1.15
#1     0.2         766       12.55
#6     0.1         112        1.83
#4     0.0        1195       19.58
#15  -0.07           1        0.02
#26   -0.7           1        0.02
#28   -1.8           1        0.02
#12   -2.9           1        0.02
#23   -3.5           1        0.02
#19   -7.0           1        0.02
#21  -12.9           1        0.02
#22  -13.6           1        0.02
#20  -18.7           1        0.02
#27  -22.4           1        0.02
#29  error           0        0.00

#fin = '/data/tat-qa/llm_table_opt/fromrerankingcenario4/mpnet_table_intro_embeddings_cpu_512_512_filtered.full.csv
#10   83.2           1        0.02
#16    1.2           1        0.02
#3     1.0        1650       25.38
#6     0.9         642        9.88
#0     0.8        1173       18.04
#12   0.75          18        0.28
#7     0.7         333        5.12
#1     0.6         620        9.54
#5     0.5         318        4.89
#2     0.4         181        2.78
#11    0.3          13        0.20
#9    0.25           3        0.05
#4     0.2         188        2.89
#13    0.1          17        0.26
#8     0.0        1339       20.60
#14   -0.3           1        0.02
#15   -1.2           1        0.02
#17  error           2        0.03