# Chat With PDF

## How does it work?

This app uses LangChain with a Retrieval-Augmented Generation (RAG) pipeline to allow users to ask questions about a collection of research papers in PDF format. It stores document chunks in a vector store (FAISS), uses OpenAI (or Together AI) to generate answers, and maintains conversational memory.

## How to run

1. Clone repo and place PDFs in `data/`
2. Add .env file attached via email to root directory.
3. Run ingestion: `python app/ingest.py`

```bash
docker-compose run backend python app/ingest.py
```

3. Start with Docker:

```bash
docker-compose up --build
```
