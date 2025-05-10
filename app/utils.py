import os
from langchain.document_loaders import PyPDFLoader
from typing import List

def load_pdfs_from_directory(directory: str) -> List:
    """
    Loads all PDF files from a given directory and returns LangChain documents.
    """
    documents = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(directory, filename)
            print(f"Loading PDF: {path}")
            loader = PyPDFLoader(path)
            docs = loader.load()
            documents.extend(docs)
    return documents
