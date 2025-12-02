from langchain_ollama import ChatOllama
from src.config import OLLAMA_MODEL

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.2,
        )
    return _llm
