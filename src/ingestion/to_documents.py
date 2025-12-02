# src/ingestion/to_documents.py

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.chunk_schema import Chunk


def _merge_titles_with_body(chunks: List[Chunk]) -> List[Chunk]:
    """
    Merge a title with ALL following text chunks from the same section until another title appears.
    This ensures sections like 'Objectives' include their entire content.
    """
    merged: List[Chunk] = []
    active_title: Chunk | None = None
    buffer_text = []

    chunks_sorted = sorted(
        chunks,
        key=lambda c: (c.file_name, c.page_number or 0, c.id),
    )

    for ch in chunks_sorted:
        category = ch.extra.get("category") if ch.extra else None

        # If this is a title — flush previous and start a new section
        if ch.modality == "text" and category == "Title":
            if active_title:
                merged.append(
                    Chunk(
                        id=active_title.id,
                        content=f"{active_title.content.strip()}\n\n" + "\n".join(buffer_text),
                        modality="text",
                        file_name=active_title.file_name,
                        file_type=active_title.file_type,
                        page_number=active_title.page_number,
                        extra=active_title.extra,
                    )
                )
            active_title = ch
            buffer_text = []
            continue

        # If we have an active title, keep adding text under it
        if active_title and ch.modality == "text":
            buffer_text.append(ch.content.strip())
            continue

        # Otherwise, regular chunk
        merged.append(ch)

    # Final flush if last section had content
    if active_title:
        merged.append(
            Chunk(
                id=active_title.id,
                content=f"{active_title.content.strip()}\n\n" + "\n".join(buffer_text),
                modality="text",
                file_name=active_title.file_name,
                file_type=active_title.file_type,
                page_number=active_title.page_number,
                extra=active_title.extra,
            )
        )

    return merged



def chunks_to_documents(chunks: List[Chunk]) -> List[Document]:
    """
    Convert merged chunks into LangChain Documents.
    Uses reasonable chunk size for good retrieval.
    """
    merged_chunks = _merge_titles_with_body(chunks)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,    # smaller chunks -> more precise retrieval
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: List[Document] = []
    for ch in merged_chunks:
        # We split text and table content; image captions usually short
        parts = text_splitter.split_text(ch.content)

        for part in parts:
            docs.append(
                Document(
                    page_content=part,
                    metadata={
                        "file_name": ch.file_name,
                        "file_type": ch.file_type,
                        "page_number": ch.page_number,
                        "modality": ch.modality,  # "text" / "table" / "image"
                    },
                )
            )

    return docs
