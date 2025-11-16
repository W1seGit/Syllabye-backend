from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any

import math

from pymongo.database import Database
from langchain_openai import OpenAIEmbeddings

from .database import get_next_id

try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
except ImportError:  # pragma: no cover - optional dependency
    DocumentConverter = None  # type: ignore
    HybridChunker = None  # type: ignore


_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def _get_chunks_from_pdf(path: str) -> List[str]:
    if DocumentConverter is None or HybridChunker is None:
        raise RuntimeError(
            "Docling is not installed. Please add 'docling' to requirements.txt and install it."
        )
    converter = DocumentConverter()
    result = converter.convert(path)
    dl_doc = result.document

    chunker = HybridChunker()
    chunks = [chunk.text for chunk in chunker.chunk(dl_doc=dl_doc)]
    return [c for c in chunks if c.strip()]


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    return _embeddings.embed_documents(texts)


def search_syllabus_chunks(
    db: Database,
    *,
    owner_id: int,
    class_id: int | None = None,
    syllabus_id: int | None = None,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Vector search over syllabus_chunks for a specific user and class.

    Optionally filter to a single syllabus_id. Returns a list of chunk docs
    sorted by similarity, each containing at least "text" and "score".
    """

    query = (query or "").strip()
    if not query:
        return []

    # Embed the query
    query_emb = _embed_texts([query])[0]

    collection = db["syllabus_chunks"]
    filter_doc: Dict[str, Any] = {"owner_id": owner_id}
    if class_id is not None:
        filter_doc["class_id"] = class_id
    if syllabus_id is not None:
        filter_doc["syllabus_id"] = syllabus_id

    candidates = list(collection.find(filter_doc))
    if not candidates:
        return []

    def _cosine(a: List[float], b: List[float]) -> float:
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    scored: List[Dict[str, Any]] = []
    for doc in candidates:
        emb = doc.get("embedding")
        if not isinstance(emb, list):
            continue
        score = _cosine(query_emb, emb)
        doc_with_score = dict(doc)
        doc_with_score["score"] = score
        scored.append(doc_with_score)

    scored.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return scored[: max(1, top_k)]


def index_syllabus_text(
    db: Database,
    *,
    owner_id: int,
    class_id: int,
    syllabus_id: int,
    text: str | None,
) -> None:
    """Index raw syllabus text into a vector collection.

    For now, we treat the entire text as a single chunk. If needed we can add
    more sophisticated chunking later.
    """

    chunks: List[str] = []
    if text:
        stripped = text.strip()
        if stripped:
            chunks.append(stripped)

    _index_chunks(
        db,
        owner_id=owner_id,
        class_id=class_id,
        syllabus_id=syllabus_id,
        source_type="syllabus_text",
        chunks=chunks,
    )


def index_syllabus_pdf(
    db: Database,
    *,
    owner_id: int,
    class_id: int,
    syllabus_id: int,
    pdf_path: str | None,
) -> None:
    """Index a syllabus PDF file using Docling for chunking."""

    if not pdf_path:
        return

    chunks = _get_chunks_from_pdf(pdf_path)
    _index_chunks(
        db,
        owner_id=owner_id,
        class_id=class_id,
        syllabus_id=syllabus_id,
        source_type="syllabus_pdf",
        chunks=chunks,
    )


def _index_chunks(
    db: Database,
    *,
    owner_id: int,
    class_id: int,
    syllabus_id: int,
    source_type: str,
    chunks: List[str],
) -> None:
    """Helper to upsert chunk embeddings for a syllabus.

    This clears any existing chunks for the given syllabus/source_type and
    inserts fresh ones.
    """

    collection = db["syllabus_chunks"]

    # Remove old chunks for this syllabus/source type
    collection.delete_many(
        {
            "owner_id": owner_id,
            "class_id": class_id,
            "syllabus_id": syllabus_id,
            "source_type": source_type,
        }
    )

    if not chunks:
        return

    embeddings = _embed_texts(chunks)
    now = datetime.utcnow()

    docs = []
    for idx, (text, emb) in enumerate(zip(chunks, embeddings)):
        doc_id = get_next_id("syllabus_chunks")
        docs.append(
            {
                "_id": doc_id,
                "id": doc_id,
                "owner_id": owner_id,
                "class_id": class_id,
                "syllabus_id": syllabus_id,
                "source_type": source_type,
                "chunk_index": idx,
                "text": text,
                "embedding": emb,
                "created_at": now,
            }
        )

    if docs:
        collection.insert_many(docs)
