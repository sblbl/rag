# RAG 
## ollama requirements
```
ollama pull mistral:latest
ollama pull nomic-embed-text:latest
```

## run the app
```
source env/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```