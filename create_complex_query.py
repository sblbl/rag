import argparse
import os
import logging
from langchain_ollama import OllamaLLM
from pydantic import TypedDict, Field
from typing import Optional, List

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("query_text", type=str, help="The query text.")
	args = parser.parse_args()
	query_text = args.query_text
	create_complex_query(query_text)

def create_complex_query(query_text: str):
	base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
	logging.info(f"Using Ollama base URL: {base_url}")
	model = OllamaLLM(
		model="mistral:latest",
		base_url="http://0.0.0.0:11434"
	)
	
	response_text = model.invoke(query_text)
	print(response_text)


if __name__ == "__main__":
	main()
	