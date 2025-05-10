from langchain.memory import ConversationBufferMemory


def get_memory():
    """
    Returns a single-instance conversation memory object.
    This stores past interactions to enable contextual follow-ups.
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return memory
