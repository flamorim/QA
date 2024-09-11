import os
import pandas as pd
import re

# Função para limpar o nome das colunas, removendo caracteres especiais
def limpar_nome_colunas(coluna):
    # Mantém apenas letras, números e underscores
    return re.sub(r'[^a-zA-Z0-9_]', '', coluna)


def sanitized_column_name(col_name):

    col_name = re.sub(r"\W+","-",col_name)
    if col_name.endswith('-'):
        col_name = col_name[:-1]
    if col_name.startswith('-'):
        col_name = col_name[1:]
    return col_name

# Função para processar todos os arquivos CSV em um diretório
def processar_csvs_diretorio(diretorio):
    idx = 0
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith('.csv'):
            caminho_arquivo = os.path.join(diretorio, arquivo)
            print(f'Processando {caminho_arquivo}')

            # Lê o arquivo CSV
            df = pd.read_csv(caminho_arquivo)

            # Limpa os nomes das colunas
            df.columns = [sanitized_column_name(coluna) for coluna in df.columns]
            print(df.columns)
            new_file = caminho_arquivo.replace('csv-original','csv')
            # Salva o arquivo de volta
            df.to_csv(new_file, index=False)
            print(f'Colunas processadas e arquivo salvo: {new_file}')
            idx +=1
    print(idx)

# Caminho do diretório com arquivos CSV
diretorio_csv = '/data/tat-qa/csv-original'

# Processa todos os arquivos CSV no diretório
processar_csvs_diretorio(diretorio_csv)