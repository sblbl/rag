import argparse
import os
import logging
import json
from langchain_ollama import OllamaLLM
from pydantic import Field, BaseModel
from typing import List

class ComplexQuerySchema(BaseModel):
    queries: List[str] = Field(default_factory=lambda: ["", "", ""], 
                              min_items=3, 
                              max_items=3, 
                              description="A list of 3 related and more specific questions related to the original query.")

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
    
    prompt = f"""
    Given the query "{query_text}", generate 3 related and more specific questions.
    Return ONLY a JSON object with this exact structure:
    {{
        "queries": [
            "question 1",
            "question 2",
            "question 3"
        ]
    }}
    """
    
    response_text = model.invoke(prompt)
    
    try:
        # Try to parse the response as JSON
        response_json = json.loads(response_text)
        complex_query = ComplexQuerySchema.model_validate(response_json)
        print(f"Complex query: {complex_query}")
        return [query for query in complex_query.queries]
    except json.JSONDecodeError as e:
        print(f"Error: LLM response was not valid JSON: {response_text}")
        # As a fallback, try to parse the text response into our desired format
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        queries = [line.split('. ', 1)[-1] for line in lines[:3]]
        if len(queries) == 3:
            complex_query = ComplexQuerySchema(queries=queries)
            print(complex_query)
            return [query for query in complex_query.queries]
        else:
            raise ValueError(f"Could not parse response into 3 questions: {response_text}")

if __name__ == "__main__":
    main()