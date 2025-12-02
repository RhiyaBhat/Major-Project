# src/ingestion/extract.py

import uuid
from pathlib import Path
from typing import List

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx

from src.ingestion.chunk_schema import Chunk


def make_chunk(element, modality, path, file_type):
    meta = element.metadata or {}
    text = getattr(element, "text", "").strip()
    page = getattr(meta, "page_number", None)

    # images
    if modality == "image":
        caption = text or getattr(meta, "alt_text", "") or "Image"
        content = caption
    else:
        content = text

    if not content:
        return None

    return Chunk(
        id=str(uuid.uuid4()),
        content=content,
        modality=modality,
        file_name=Path(path).name,
        file_type=file_type,
        page_number=page,
        extra={
            "category": getattr(element, "category", None),
            "coordinates": getattr(meta, "coordinates", None),
            "image_path": getattr(meta, "image_path", None),
        },
    )


def extract_pdf(path: str) -> List[Chunk]:
    """Uses full Unstructured extraction: text, tables, and images."""
    elements = partition_pdf(
        filename=path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        languages=["eng"],
        strategy="hi_res",
    )

    chunks = []

    for e in elements:
        cat = getattr(e, "category", "")
        if cat in ["NarrativeText", "Title", "ListItem"]:
            modality = "text"
        elif cat == "Table":
            modality = "table"
        elif cat == "Image":
            modality = "image"
        else:
            continue

        chunk = make_chunk(e, modality, path, "pdf")
        if chunk:
            chunks.append(chunk)

    return chunks


def extract_pptx(path: str) -> List[Chunk]:
    elements = partition_pptx(path, extract_images_in_pptx=True)
    chunks = []

    for e in elements:
        cat = getattr(e, "category", "")
        modality = (
            "text" if cat in ["NarrativeText", "Title", "ListItem"]
            else "table" if cat == "Table"
            else "image" if cat == "Image"
            else None
        )
        if not modality:
            continue

        chunk = make_chunk(e, modality, path, "pptx")
        if chunk:
            chunks.append(chunk)

    return chunks


def extract_from_files(file_paths: List[str]) -> List[Chunk]:
    final = []

    for path in file_paths:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            final.extend(extract_pdf(path))
        elif ext in [".pptx", ".ppt"]:
            final.extend(extract_pptx(path))
        else:
            print(f"[WARN] Unsupported file type: {path}")

    return final
