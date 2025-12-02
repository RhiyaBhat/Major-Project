# src/ingestion/chunk_schema.py

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Chunk:
    id: str
    content: str            # text we will embed
    modality: str           # "text" | "table" | "image"
    file_name: str
    file_type: str          # "pdf" | "pptx"
    page_number: Optional[int]
    extra: Dict[str, Any]
