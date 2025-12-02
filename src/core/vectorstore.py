from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.core.embeddings import get_embeddings
from src.config import CHROMA_DIR


def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name="llamachain_docs",
    )


def add_documents(docs: list[Document]) -> int:
    vs = get_vectorstore()
    vs.add_documents(docs)
    return len(docs)
