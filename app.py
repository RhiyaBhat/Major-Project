import os
import sys
from typing import List

import streamlit as st

# Make sure Python can see the src/ package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Import your project modules
from src.ingestion.extract import extract_from_files
from src.ingestion.to_documents import chunks_to_documents
from src.core.vectorstore import add_documents
from src.rag.rag_chain import build_rag_chain


# ---------- Helpers ----------

def save_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    """
    Save uploaded files to data/raw and return their file paths.
    """
    raw_dir = os.path.join(CURRENT_DIR, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    paths: List[str] = []
    for f in uploaded_files:
        dest_path = os.path.join(raw_dir, f.name)
        with open(dest_path, "wb") as out:
            out.write(f.read())
        paths.append(dest_path)

    return paths


def ensure_chain():
    """Initialize RAG chain + chat history in session_state if not present."""
    if "chain" not in st.session_state:
        st.session_state.chain = build_rag_chain()
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (question, answer, sources)


# ---------- Streamlit UI ----------

st.set_page_config(
    page_title="LlamaChain — Docs Chat (Offline)",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 LlamaChain — Chat with Your Documents (Offline)")


# ===== Sidebar: Ingestion =====
with st.sidebar:
    st.header("📥 Ingest / Index Documents")

    st.markdown(
        "Upload PDFs / PPT / PPTX here.\n\n"
        "They will be parsed with Unstructured (text + tables + images), "
        "embedded, and stored in the local ChromaDB."
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "ppt", "pptx"],
        accept_multiple_files=True,
    )

    ingest_clicked = st.button("🚀 Process & Index", type="primary")

    if ingest_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one file before processing.")
        else:
            with st.spinner("Extracting with Unstructured and storing embeddings in Chroma..."):
                # 1) Save to disk
                paths = save_uploaded_files(uploaded_files)

                # 2) Extract raw chunks (text, tables, images) from PDFs/PPTX
                chunks = extract_from_files(paths)

                # 3) Convert to LangChain Documents (with merged titles + chunking)
                docs = chunks_to_documents(chunks)

                # 4) Add to vector store
                num = add_documents(docs)

                # 5) Reset RAG chain so it uses the updated DB
                st.session_state.pop("chain", None)
                st.session_state.pop("history", None)

            st.success(f"✅ Indexed {num} chunks from {len(uploaded_files)} file(s).")
            st.info("You can now ask questions about these documents in the main chat area.")


st.divider()

# ===== Main: Chat Interface =====
ensure_chain()

st.subheader("💬 Chat with your ingested documents")

query = st.chat_input("Ask a question using the ingested PDFs/PPTX...")

if query:
    rag = st.session_state.chain

    with st.spinner("Thinking with local Llama..."):
        result = rag(query)

    answer = result["answer"]
    sources = result.get("source_documents", [])

    # Save in history
    st.session_state.history.append((query, answer, sources))


# Render chat history
for idx, (q, a, srcs) in enumerate(st.session_state.history):
    st.chat_message("user").write(q)

    with st.chat_message("assistant"):
        st.write(a)

        if srcs:
            with st.expander("📚 Sources", expanded=False):
                for i, doc in enumerate(srcs, start=1):
                    meta = doc.metadata or {}
                    st.markdown(
                        f"**Source {i}:** "
                        f"`{meta.get('file_name', '?')}`, "
                        f"page `{meta.get('page_number', '?')}`, "
                        f"modality `{meta.get('modality', '?')}`"
                    )
                    # Show a preview of the chunk
                    preview = doc.page_content[:500]
                    st.write(preview + ("…" if len(doc.page_content) > 500 else ""))
                    st.markdown("---")
