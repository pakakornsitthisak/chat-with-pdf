from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


from app.memory import get_memory
import os

VECTORSTORE_DIR = "vectorstore"  # or use env/config
TEMPERATURE = 0.0

def load_vectorstore():
    """
    Loads the FAISS vector store from local directory.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True  # ✅ explicitly allow it
    )

    return vectorstore.as_retriever()

def get_rag_chain():
    """
    Creates and returns a ConversationalRetrievalChain with memory.
    """
    retriever = load_vectorstore()
    memory = get_memory()
    llm = ChatOpenAI(temperature=TEMPERATURE)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )

    return qa_chain, memory
