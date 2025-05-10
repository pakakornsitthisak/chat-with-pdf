from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from utils import load_pdfs_from_directory
import os
from dotenv import load_dotenv
import openai

# Directories
PDF_DIR = "data"  # Replace with your folder where PDFs are stored
VECTORSTORE_DIR = "vectorstore"  # Folder to store FAISS index

def main():
    # Step 1: Load PDF documents
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("[INFO] Loading PDF documents...")
    documents = load_pdfs_from_directory(PDF_DIR)

    # Step 2: Split documents into smaller chunks for embedding
    print(f"[INFO] Loaded {len(documents)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    print(f"[INFO] Total chunks: {len(chunks)}. Generating embeddings...")
    embeddings = OpenAIEmbeddings()

    # Step 3: Create FAISS vector store and store the embeddings
    print("[INFO] Creating vector store (FAISS)...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTORSTORE_DIR)

    print(f"[SUCCESS] Vector store saved to: {VECTORSTORE_DIR}")

if __name__ == "__main__":
    main()
