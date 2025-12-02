import os
from pathlib import Path

import fitz  # PyMuPDF
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 🔁 CHANGE THIS TO A REAL PDF YOU HAVE
TEST_PDF_PATH = r"C:\Users\Admin\OneDrive\Desktop\Major Project\Chat with Pdfs\Major Project Synopsis - report (pdf).pdf"


def extract_docs_from_pdf_simple(pdf_path: str):
    print(f"[INFO] Extracting (simple) from: {pdf_path}")
    doc = fitz.open(pdf_path)

    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if not text.strip():
            continue

        # very simple chunking: split by double newline into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for para in paragraphs:
            chunks.append(
                Document(
                    page_content=para,
                    metadata={
                        "page_number": page_num + 1,
                        "file_name": os.path.basename(pdf_path),
                        "source": "pymupdf_simple",
                    },
                )
            )

    print(f"[INFO] Extracted {len(chunks)} text chunks.")
    if chunks:
        print("[SAMPLE CHUNK]")
        print("Page:", chunks[0].metadata["page_number"])
        print("Text:", chunks[0].page_content[:300], "...\n")
    return chunks


def build_vectorstore(docs):
    print("[INFO] Creating embeddings & Chroma vector store...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vs = Chroma(
        collection_name="test_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_test",
    )

    vs.add_documents(docs)
    vs.persist()
    print("[INFO] Stored docs in Chroma (./chroma_test).")
    return vs


def test_query(vs):
    print("\n[INFO] Testing retrieval...")
    query = "What is this document mainly about?"
    print("Query:", query)

    results = vs.similarity_search(query, k=3)

    print(f"[INFO] Got {len(results)} results.\n")
    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print("Page:", doc.metadata.get("page_number"))
        print("File:", doc.metadata.get("file_name"))
        print("Text:", doc.page_content[:300], "...\n")


if __name__ == "__main__":
    if not Path(TEST_PDF_PATH).exists():
        print(f"[ERROR] TEST_PDF_PATH does not exist:\n{TEST_PDF_PATH}")
        print("➡ Change TEST_PDF_PATH at the top of test_pipeline.py to a real PDF.")
    else:
        docs = extract_docs_from_pdf_simple(TEST_PDF_PATH)
        if docs:
            vs = build_vectorstore(docs)
            test_query(vs)
        else:
            print("[WARN] No chunks extracted from this PDF.")
