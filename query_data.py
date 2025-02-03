

import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import os
from get_embedding_function import get_embedding_function
from create_complex_query import create_complex_query
import logging

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

model = None
embedding_function = None
db = None

def main():
	# Create CLI.
	parser = argparse.ArgumentParser()
	parser.add_argument("query_text", type=str, help="The query text.")
	args = parser.parse_args()
	query_text = args.query_text
	query_rag(query_text)

def query_db(query_text: str):
	# Search the DB.
	results = db.similarity_search_with_score(query_text, k=5)
	context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
	prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
	prompt = prompt_template.format(context=context_text, question=query_text)
	#response_text = model.invoke(query_text)
	response_text = model.invoke(prompt)
	"""
	print("\nAI Response:")
	print(response_text)
	print("\n\n\n")
	print("\nSources Used:")
	for i, (doc, score) in enumerate(results, 1):
		print(f"\nSource {i}:")
		print(f"- ID: {doc.metadata.get('id', 'Unknown')}")
		print(f"- Similarity Score: {score:.2f}")
		print(f"- Content:")
		print("---")
		print(doc.page_content)
		print("---")
		print("\n")
	"""
	return {
		"question": query_text,
		"text": response_text,
		"sources": [{"id": doc.metadata.get("id", None), "page_content": doc.page_content, "score": score} for doc, score in results]
	}

def query_rag(query_text: str):
	global model
	global embedding_function
	global db
	base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
	logging.info(f"Using Ollama base URL: {base_url}")
	model = OllamaLLM(
		model="mistral:latest",
		base_url="http://0.0.0.0:11434"
	)
	embedding_function = get_embedding_function()
	db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
	try:
		complex_query = create_complex_query(query_text)
		all_questions = complex_query
		all_questions.insert(0, query_text)
		print(f"All questions: {all_questions}")

		all_responses = []

		for i, question in enumerate(all_questions):
			print(f"Answering question {i+1}: {question}")
			response = query_db(question)
			all_responses.append(response)

		print(f"All responses: {all_responses}")
	except Exception as e:
		logging.error(f"Ollama error: {str(e)}")
		raise

if __name__ == "__main__":
	main()