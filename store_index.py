from src.helper import repo_ignestion, load_repo, text_splitter, load_embedding
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import os

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

url = ""

# repo_ignestion(url)

documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = load_embedding()




## storing vectors in chromadb

vectorsdb = Chroma.from_documents(texts, embedding=embedding, persist_directory='.DB')
vectordb.persist()



