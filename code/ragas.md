pip install ragas 
Collecting ragas
  Downloading ragas-0.1.9-py3-none-any.whl (86 kB)
     |████████████████████████████████| 86 kB 1.1 MB/s 
Collecting pysbd>=0.3.4
  Downloading pysbd-0.3.4-py3-none-any.whl (71 kB)
     |████████████████████████████████| 71 kB 1.7 MB/s 
Requirement already satisfied: appdirs in /nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/home/.local/lib/python3.8/site-packages (from ragas) (1.4.4)
Requirement already satisfied: langchain-community in /nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/home/.local/lib/python3.8/site-packages (from ragas) (0.0.19)
Requirement already satisfied: langchain in /nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/home/.local/lib/python3.8/site-packages (from ragas) (0.1.5)
Requirement already satisfied: tiktoken in /usr/local/lib/python3.8/dist-packages (from ragas) (0.5.2)
Collecting openai>1
  Downloading openai-1.35.7-py3-none-any.whl (327 kB)
     |████████████████████████████████| 327 kB 3.0 MB/s 
Requirement already satisfied: nest-asyncio in /usr/local/lib/python3.8/dist-packages (from ragas) (1.5.8)
Collecting langchain-openai
  Downloading langchain_openai-0.1.13-py3-none-any.whl (45 kB)
     |████████████████████████████████| 45 kB 2.3 MB/s 
Requirement already satisfied: numpy in /usr/local/lib/python3.8/dist-packages (from ragas) (1.23.5)
Requirement already satisfied: langchain-core in /nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/home/.local/lib/python3.8/site-packages (from ragas) (0.1.21)
Requirement already satisfied: datasets in /nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/home/.local/lib/python3.8/site-packages (from ragas) (2.18.0)
Requirement already satisfied: dataclasses-json<0.7,>=0.5.7 in /usr/local/lib/python3.8/dist-packages (from langchain-community->ragas) (0.6.3)
Requirement already satisfied: PyYAML>=5.3 in /usr/local/lib/python3.8/dist-packages (from langchain-community->ragas) (6.0.1)
Requirement already satisfied: requests<3,>=2 in /usr/local/lib/python3.8/dist-packages (from langchain-community->ragas) (2.31.0)
Requirement already satisfied: langsmith<0.1,>=0.0.83 in /nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/home/.local/lib/python3.8/site-packages (from langchain-community->ragas) (0.0.86)
Requirement already satisfied: SQLAlchemy<3,>=1.4 in /usr/local/lib/python3.8/dist-packages (from langchain-community->ragas) (2.0.23)
Requirement already satisfied: tenacity<9.0.0,>=8.1.0 in /usr/local/lib/python3.8/dist-packages (from langchain-community->ragas) (8.2.3)
Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /usr/local/lib/python3.8/dist-packages (from langchain-community->ragas) (3.9.1)
Requirement already satisfied: pydantic<3,>=1 in /usr/local/lib/python3.8/dist-packages (from langchain->ragas) (2.5.2)
Requirement already satisfied: jsonpatch<2.0,>=1.33 in /usr/local/lib/python3.8/dist-packages (from langchain->ragas) (1.33)
Requirement already satisfied: async-timeout<5.0.0,>=4.0.0; python_version < "3.11" in /usr/local/lib/python3.8/dist-packages (from langchain->ragas) (4.0.3)
Requirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.8/dist-packages (from tiktoken->ragas) (2023.10.3)
Collecting distro<2,>=1.7.0
  Using cached distro-1.9.0-py3-none-any.whl (20 kB)
Requirement already satisfied: anyio<5,>=3.5.0 in /usr/local/lib/python3.8/dist-packages (from openai>1->ragas) (3.7.1)
Requirement already satisfied: tqdm>4 in /usr/local/lib/python3.8/dist-packages (from openai>1->ragas) (4.66.1)
Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.8/dist-packages (from openai>1->ragas) (0.25.2)
Requirement already satisfied: typing-extensions<5,>=4.7 in /usr/local/lib/python3.8/dist-packages (from openai>1->ragas) (4.9.0)
Requirement already satisfied: sniffio in /usr/local/lib/python3.8/dist-packages (from openai>1->ragas) (1.3.0)
Requirement already satisfied: packaging<24.0,>=23.2 in /usr/local/lib/python3.8/dist-packages (from langchain-core->ragas) (23.2)
Requirement already satisfied: fsspec[http]<=2024.2.0,>=2023.1.0 in /usr/local/lib/python3.8/dist-packages (from datasets->ragas) (2023.12.2)
Requirement already satisfied: filelock in /usr/local/lib/python3.8/dist-packages (from datasets->ragas) (3.13.1)
Requirement already satisfied: dill<0.3.9,>=0.3.0 in /usr/local/lib/python3.8/dist-packages (from datasets->ragas) (0.3.7)
Requirement already satisfied: pyarrow-hotfix in /nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/home/.local/lib/python3.8/site-packages (from datasets->ragas) (0.6)
Requirement already satisfied: huggingface-hub>=0.19.4 in /usr/local/lib/python3.8/dist-packages (from datasets->ragas) (0.19.4)
Requirement already satisfied: pyarrow>=12.0.0 in /usr/local/lib/python3.8/dist-packages (from datasets->ragas) (14.0.1)
Requirement already satisfied: pandas in /usr/local/lib/python3.8/dist-packages (from datasets->ragas) (1.5.2)
Requirement already satisfied: multiprocess in /usr/local/lib/python3.8/dist-packages (from datasets->ragas) (0.70.15)
Requirement already satisfied: xxhash in /nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/home/.local/lib/python3.8/site-packages (from datasets->ragas) (3.4.1)
Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /usr/local/lib/python3.8/dist-packages (from dataclasses-json<0.7,>=0.5.7->langchain-community->ragas) (3.20.1)
Requirement already satisfied: typing-inspect<1,>=0.4.0 in /usr/local/lib/python3.8/dist-packages (from dataclasses-json<0.7,>=0.5.7->langchain-community->ragas) (0.9.0)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests<3,>=2->langchain-community->ragas) (3.6)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests<3,>=2->langchain-community->ragas) (2023.11.17)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.8/dist-packages (from requests<3,>=2->langchain-community->ragas) (3.3.2)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests<3,>=2->langchain-community->ragas) (2.1.0)
Requirement already satisfied: greenlet!=0.4.17; platform_machine == "aarch64" or (platform_machine == "ppc64le" or (platform_machine == "x86_64" or (platform_machine == "amd64" or (platform_machine == "AMD64" or (platform_machine == "win32" or platform_machine == "WIN32"))))) in /usr/local/lib/python3.8/dist-packages (from SQLAlchemy<3,>=1.4->langchain-community->ragas) (3.0.2)
Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.8/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community->ragas) (1.4.0)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.8/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community->ragas) (23.1.0)
Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.8/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community->ragas) (1.9.4)
Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.8/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community->ragas) (1.3.1)
Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.8/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community->ragas) (6.0.4)
Requirement already satisfied: annotated-types>=0.4.0 in /usr/local/lib/python3.8/dist-packages (from pydantic<3,>=1->langchain->ragas) (0.6.0)
Requirement already satisfied: pydantic-core==2.14.5 in /usr/local/lib/python3.8/dist-packages (from pydantic<3,>=1->langchain->ragas) (2.14.5)
Requirement already satisfied: jsonpointer>=1.9 in /usr/local/lib/python3.8/dist-packages (from jsonpatch<2.0,>=1.33->langchain->ragas) (2.4)
Requirement already satisfied: exceptiongroup; python_version < "3.11" in /usr/local/lib/python3.8/dist-packages (from anyio<5,>=3.5.0->openai>1->ragas) (1.2.0)
Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.8/dist-packages (from httpx<1,>=0.23.0->openai>1->ragas) (1.0.2)
Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.8/dist-packages (from pandas->datasets->ragas) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.8/dist-packages (from pandas->datasets->ragas) (2023.3.post1)
Requirement already satisfied: mypy-extensions>=0.3.0 in /usr/local/lib/python3.8/dist-packages (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain-community->ragas) (1.0.0)
Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.8/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai>1->ragas) (0.14.0)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.8/dist-packages (from python-dateutil>=2.8.1->pandas->datasets->ragas) (1.16.0)
ERROR: langchain-openai 0.1.13 has requirement langchain-core<0.3,>=0.2.2, but you'll have langchain-core 0.1.21 which is incompatible.
ERROR: langchain-openai 0.1.13 has requirement tiktoken<1,>=0.7, but you'll have tiktoken 0.5.2 which is incompatible.
Installing collected packages: pysbd, distro, openai, langchain-openai, ragas
  WARNING: The script distro is installed in '/nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/home/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script openai is installed in '/nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/home/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
Successfully installed distro-1.9.0 langchain-openai-0.1.13 openai-1.35.7 pysbd-0.3.4 ragas-0.1.9
Singularity> 



Singularity> pip install Dataset
Collecting Dataset
  Downloading dataset-1.6.2-py2.py3-none-any.whl (18 kB)
Collecting sqlalchemy<2.0.0,>=1.3.2
  Downloading SQLAlchemy-1.4.52-cp38-cp38-manylinux1_x86_64.manylinux2010_x86_64.manylinux_2_12_x86_64.manylinux_2_5_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.6 MB)
     |████████████████████████████████| 1.6 MB 3.9 MB/s 
Collecting banal>=1.0.1
  Downloading banal-1.0.6-py2.py3-none-any.whl (6.1 kB)
Collecting alembic>=0.6.2
  Downloading alembic-1.13.2-py3-none-any.whl (232 kB)
     |████████████████████████████████| 232 kB 6.5 MB/s 
Requirement already satisfied: greenlet!=0.4.17; python_version >= "3" and (platform_machine == "aarch64" or (platform_machine == "ppc64le" or (platform_machine == "x86_64" or (platform_machine == "amd64" or (platform_machine == "AMD64" or (platform_machine == "win32" or platform_machine == "WIN32")))))) in /usr/local/lib/python3.8/dist-packages (from sqlalchemy<2.0.0,>=1.3.2->Dataset) (3.0.2)
Requirement already satisfied: typing-extensions>=4 in /usr/local/lib/python3.8/dist-packages (from alembic>=0.6.2->Dataset) (4.9.0)
Requirement already satisfied: importlib-resources; python_version < "3.9" in /usr/local/lib/python3.8/dist-packages (from alembic>=0.6.2->Dataset) (6.1.1)
Requirement already satisfied: importlib-metadata; python_version < "3.9" in /usr/local/lib/python3.8/dist-packages (from alembic>=0.6.2->Dataset) (6.11.0)
Collecting Mako
  Downloading Mako-1.3.5-py3-none-any.whl (78 kB)
     |████████████████████████████████| 78 kB 5.8 MB/s 
Requirement already satisfied: zipp>=3.1.0; python_version < "3.10" in /usr/local/lib/python3.8/dist-packages (from importlib-resources; python_version < "3.9"->alembic>=0.6.2->Dataset) (3.17.0)
Requirement already satisfied: MarkupSafe>=0.9.2 in /usr/local/lib/python3.8/dist-packages (from Mako->alembic>=0.6.2->Dataset) (2.1.3)
Installing collected packages: sqlalchemy, banal, Mako, alembic, Dataset
  WARNING: The script mako-render is installed in '/nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/home/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script alembic is installed in '/nethome/projetos30/busca_semantica/buscaict/BigOil/users/flavio.amorim/home/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
Successfully installed Dataset-1.6.2 Mako-1.3.5 alembic-1.13.2 banal-1.0.6 sqlalchemy-1.4.52
Singularity> 


verificar:
 sudo apt install libsndfile1