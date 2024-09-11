import json
import re

# Lista para armazenar os dicionários
data_list = []

# Abrir o arquivo input.jsonl para leitura

def old():
        with open('/data/wikitablequestions/data/pristine-unseen-tables.examples','r') as fin:
            # Iterar sobre as linhas do arquivo
            for input_string in fin:


                # Expressões regulares para extrair os dados
                utterance_pattern = r'utterance "(.*?)"'
                description_pattern = r'description "(.*?)"'
                table_pattern = r'TableKnowledgeGraph (.*?)\)'
                id_patern = r'id (.*?)\)'
        

                # Extrair os dados usando expressões regulares
                utterance_match = re.search(utterance_pattern, input_string)
                description_match = re.search(description_pattern, input_string)
                table_match = re.search(table_pattern, input_string)
                id_match = re.search(id_patern, input_string)

                # Verificar se todas as informações foram encontradas
                if utterance_match and description_match and table_match:
                    # Criar o dicionário com os dados extraídos
                    data_dict = {
                        'id':id_match.group(1),
                        'question': utterance_match.group(1),
                        'answer': description_match.group(1),
                        'table': f'/data/wikitablequestions/{table_match.group(1)}'
                    }
                    print(data_dict)
                    data_list.append(data_dict)
                else:
                    print("Não foi possível extrair todas as informações necessárias.")



        # Imprimir a lista de dicionários
        print(len(data_list))

        # Salvar a lista de dicionários em um arquivo JSON
        with open('/data/wikitablequestions/dataset_unseen-tables.json', 'w') as json_file:
            json.dump(data_list, json_file)

def create_dataset_table_text():

        new_dataset_list = []
        dataset_raw_list = get_dataset('/data/wikitablequestions/dataset_unseen-tables.json',device)

        for data_raw in dataset_raw_list:
                data = data_raw.copy()
                table = data['table']
                text = getfirst_paragrapg(table)
                data['text'] = text
                new_dataset_list.append(data)
                print(count)
                print(answer_table_id)

        with open('/data/wikitablequestions/new_dataset.json', 'w') as json_file:
            json.dump(new_dataset_list, json_file)

