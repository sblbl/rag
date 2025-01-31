from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from query_data import query_rag
import uvicorn
import logging
from populate_database import load_documents, split_documents, add_embedding_prefixes, add_to_chroma
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Lifecycle event handler for database initialization"""
	try:
		logger.info("Starting database population...")
		documents = load_documents()
		chunks = split_documents(documents)
		chunks_with_prefixes = add_embedding_prefixes(chunks)
		add_to_chroma(chunks_with_prefixes)
		logger.info("Database population completed successfully")
	except Exception as e:
		logger.error(f"Error populating database: {e}")
		raise e
	yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create a templates directory and mount it
templates = Jinja2Templates(directory="templates")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
	return templates.TemplateResponse(
		"index.html", 
		{"request": request, "response": None}
	)

@app.post("/query", response_class=HTMLResponse)
async def query(request: Request, query: str = Form(...)):
	try:
		response = query_rag(query)
		return JSONResponse({
			"response": response["text"],
			"sources": response["sources"],
			"query": query
		})
	except Exception as e:
		logger.error(f"Error querying RAG: {e}")
		response = {"text": "An error occurred while processing your request.", "sources": []}
		return templates.TemplateResponse(
			"index.html", 
			{"request": request, "response": response["text"], "sources": response["sources"], "query": query}
		)

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
	return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
	return JSONResponse({"detail": exc.errors()}, status_code=400)

if __name__ == "__main__":
	uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)