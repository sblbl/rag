#from langchain_ollama import OllamaEmbeddings
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
import os


def get_embedding_function():
	base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
	#embeddings = OllamaEmbeddings(model="mxbai-embed-large")
	embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)

	return embeddings