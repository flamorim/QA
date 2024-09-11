import requests
from bs4 import BeautifulSoup
  
        
def get_first_paragraph(url):        
        
        # Faz a solicitação HTTP para a URL
        response = requests.get(url)
        response.raise_for_status()  # Levanta um erro se a solicitação falhar

        # Analisa o conteúdo HTML da página
        soup = BeautifulSoup(response.content, 'html.parser')

        # Encontra o primeiro parágrafo no conteúdo principal da página
        content = soup.find('div', {'class': 'mw-parser-output'})
        #first_paragraph = content.find('p').get_text()
        paragraphs = content.find_all('p')

        # Itera sobre os parágrafos até encontrar um não vazio
        for paragraph in paragraphs:
            text = paragraph.get_text().strip()
            if text:  # Verifica se o parágrafo não está vazio
                print(text)
                break



# Exemplo de uso
#url = input("Digite a URL de uma página da Wikipedia: ")
#url = 'https://en.wikipedia.org/wiki?action=render&curid=291520&oldid=599434140'
#irst_paragraph = get_first_paragraph(url)
#print("Primeiro parágrafo:", first_paragraph)
import pandas as pd

df = pd.read_csv('/data/tat-qa/retriever/mpnet_table_embeddings_cpu_512_512.csv', sep=',')
dff['intro'] = df['top100_table_intro'].apply(lambda x: eval(x)[0])

dff.to_csv('/data/tat-qa/retriever/intro.csv',index=False)
