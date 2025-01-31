import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


CHROMA_PATH = "/Users/marta/Desktop/raga/chroma"
DATA_PATH = "/Users/marta/Desktop/raga/data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    chunks_with_prefixes = add_embedding_prefixes(chunks)
    add_to_chroma(chunks_with_prefixes)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_embedding_prefixes(chunks: list[Document]) -> list[Document]:
    """Add prefixes to document chunks to provide context for the embedding model."""
    prefixed_chunks = []
    
    for chunk in chunks:
        # Create a copy of the chunk to avoid modifying the original
        prefixed_chunk = Document(
            page_content=chunk.page_content,
            metadata=chunk.metadata.copy()
        )
        
        # Get document type and other metadata
        source = chunk.metadata.get("source", "")
        doc_type = "pdf"  # You can expand this based on file extensions
        
        # Create prefix based on document type and source
        prefix = f"[DOC_TYPE: {doc_type}] [SOURCE: {source}] "
        
        # Add semantic prefix based on content type
        # You can expand these conditions based on your document types
        if "table" in chunk.page_content.lower():
            prefix += "[CONTENT_TYPE: table] "
        elif any(term in chunk.page_content.lower() for term in ["figure", "fig.", "graph"]):
            prefix += "[CONTENT_TYPE: figure] "
        else:
            prefix += "[CONTENT_TYPE: text] "
        
        # Add the prefixed content
        prefixed_chunk.page_content = prefix + chunk.page_content
        
        # Store original content in metadata for reference
        prefixed_chunk.metadata["original_content"] = chunk.page_content
        
        prefixed_chunks.append(prefixed_chunk)
    
    return prefixed_chunks


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()