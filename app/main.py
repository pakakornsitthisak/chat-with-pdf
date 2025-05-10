from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import get_rag_chain

# Create FastAPI app instance
app = FastAPI(
    title="Chat with PDF",
    description="A simple RAG-based QA system over academic papers",
    version="1.0.0"
)

# Request schema
class Question(BaseModel):
    query: str

# Load RAG chain and memory at startup
qa_chain, memory = get_rag_chain()

@app.get("/")
def read_root():
    return {"status": "Running", "message": "Chat With PDF API is live."}

@app.post("/ask")
def ask_question(question: Question):
    """
    Accepts a user query and returns an answer based on the ingested PDFs.
    """
    try:
        response = qa_chain.run(question.query)
        return {"question": question.query, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}")

@app.post("/reset")
def reset_memory():
    """
    Clears the chat memory (single session only).
    """
    try:
        memory.clear()
        return {"status": "success", "message": "Conversation memory has been reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset memory: {e}")
