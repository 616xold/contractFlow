"""Chunking and retrieval utilities for contract text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

import hashlib
import json
import math
import re

from openai import OpenAI

from contractflow.core.pdf_utils import read_pdf_pages


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HEADING_PREFIX_RE = re.compile(r"^(section|article)\s+\d+[\s:.-]+", re.IGNORECASE)
_HEADING_NUMBERED_RE = re.compile(r"^\d+(\.\d+)*\s+\S+")
_ALL_CAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9\s\-:,().]{3,}$")
_HEADING_KEYWORDS_RE = re.compile(
    r"^(exhibit|schedule|appendix|annex|attachment|signature|signatures)\b",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    chunk_id: str
    page_num: int
    heading: Optional[str]
    chunk_text: str

    def combined_text(self) -> str:
        if self.heading:
            return f"{self.heading}\n{self.chunk_text}".strip()
        return self.chunk_text.strip()


@dataclass
class ChunkIndex:
    chunks: List[Chunk]
    term_freqs: List[Dict[str, int]]
    doc_freq: Dict[str, int]
    idf: Dict[str, float]
    doc_lens: List[int]
    avg_doc_len: float
    k1: float = 1.5
    b: float = 0.75


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float


@dataclass
class EmbeddingIndex:
    chunks: List[Chunk]
    vectors: List[List[float]]
    norms: List[float]
    model: str
    cache_path: Optional[str] = None
    cache_hit: bool = False


class ChunkRetriever(Protocol):
    backend: str
    model: Optional[str]
    cache_path: Optional[str]
    cache_hit: Optional[bool]

    def retrieve(self, query: str, *, top_k: int = 5) -> List[RetrievalHit]:
        ...


@dataclass
class BM25Retriever:
    index: ChunkIndex
    backend: str = "bm25"
    model: Optional[str] = None
    cache_path: Optional[str] = None
    cache_hit: Optional[bool] = None

    def retrieve(self, query: str, *, top_k: int = 5) -> List[RetrievalHit]:
        return retrieve(query, self.index, top_k=top_k)


@dataclass
class EmbeddingRetriever:
    index: EmbeddingIndex
    client: Optional[OpenAI] = None
    backend: str = "openai-embeddings"
    model: Optional[str] = None
    cache_path: Optional[str] = None
    cache_hit: Optional[bool] = None

    def __post_init__(self) -> None:
        if not self.model:
            self.model = self.index.model
        self.cache_path = self.index.cache_path
        self.cache_hit = self.index.cache_hit

    def retrieve(self, query: str, *, top_k: int = 5) -> List[RetrievalHit]:
        if not query.strip():
            return []
        client = self.client or OpenAI()
        query_vector = _embed_texts([query], model=self.index.model, client=client)[0]
        query_norm = _l2_norm(query_vector)
        hits: List[RetrievalHit] = []
        for chunk, vector, norm in zip(self.index.chunks, self.index.vectors, self.index.norms):
            score = _cosine_similarity(query_vector, query_norm, vector, norm)
            hits.append(RetrievalHit(chunk=chunk, score=score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: max(top_k, 0)]


def chunk_pdf(
    pdf_path: str | Path,
    *,
    max_chunk_chars: int = 2000,
    use_ocr: bool = False,
    ocr_min_chars: int = 40,
    ocr_lang: str = "eng",
    ocr_dpi: int = 200,
) -> List[Chunk]:
    """Split a PDF into page/section chunks with headings."""
    pages = read_pdf_pages(
        pdf_path,
        use_ocr=use_ocr,
        ocr_min_chars=ocr_min_chars,
        ocr_lang=ocr_lang,
        ocr_dpi=ocr_dpi,
    )
    chunks: List[Chunk] = []
    for page_num, page_text in enumerate(pages, start=1):
        page_chunks = chunk_page_text(page_text, page_num, max_chunk_chars=max_chunk_chars)
        chunks.extend(page_chunks)
    return chunks


def chunk_page_text(
    page_text: str,
    page_num: int,
    *,
    max_chunk_chars: int = 2000,
) -> List[Chunk]:
    """Split a single page into sections based on heading heuristics."""
    if not page_text or not page_text.strip():
        return []

    lines = [line.strip() for line in page_text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    has_heading = any(_is_heading(line) for line in lines)
    if not has_heading:
        body = "\n".join(lines).strip()
        return _split_long_chunk(
            page_num=page_num,
            section_index=0,
            heading=None,
            body=body,
            max_chunk_chars=max_chunk_chars,
        )

    chunks: List[Chunk] = []
    current_heading: Optional[str] = None
    current_lines: List[str] = []
    section_index = 0

    def flush() -> None:
        nonlocal section_index
        if current_heading is None and not current_lines:
            return
        body = "\n".join(current_lines).strip()
        if not body and current_heading:
            body = current_heading
        chunks.extend(
            _split_long_chunk(
                page_num=page_num,
                section_index=section_index,
                heading=current_heading,
                body=body,
                max_chunk_chars=max_chunk_chars,
            )
        )
        section_index += 1

    for line in lines:
        if _is_heading(line):
            flush()
            current_heading = line
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return chunks


def build_bm25_index(chunks: Iterable[Chunk], *, k1: float = 1.5, b: float = 0.75) -> ChunkIndex:
    """Build a BM25 index over a list of chunks."""
    chunk_list = list(chunks)
    term_freqs: List[Dict[str, int]] = []
    doc_freq: Dict[str, int] = {}
    doc_lens: List[int] = []

    for chunk in chunk_list:
        tokens = _tokenize(chunk.combined_text())
        freq: Dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        term_freqs.append(freq)
        doc_lens.append(len(tokens))
        for term in freq:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    total_docs = len(chunk_list)
    avg_doc_len = (sum(doc_lens) / total_docs) if total_docs else 0.0
    if avg_doc_len == 0.0:
        avg_doc_len = 1.0

    idf: Dict[str, float] = {}
    for term, df in doc_freq.items():
        idf[term] = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

    return ChunkIndex(
        chunks=chunk_list,
        term_freqs=term_freqs,
        doc_freq=doc_freq,
        idf=idf,
        doc_lens=doc_lens,
        avg_doc_len=avg_doc_len,
        k1=k1,
        b=b,
    )


def build_embedding_index(
    chunks: Iterable[Chunk],
    *,
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
    client: Optional[OpenAI] = None,
    cache_dir: Optional[str | Path] = None,
    cache_key: Optional[str] = None,
) -> EmbeddingIndex:
    """Build an embeddings index over a list of chunks."""
    chunk_list = list(chunks)
    texts = [chunk.combined_text() for chunk in chunk_list]
    client = client or OpenAI()
    vectors: List[List[float]] = []

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    chunk_hashes = [_hash_chunk(chunk) for chunk in chunk_list]
    cache_path = _resolve_embedding_cache_path(
        cache_dir=cache_dir,
        model=model,
        chunk_hashes=chunk_hashes,
        cache_key=cache_key,
    )
    if cache_path is not None:
        cached = _load_embedding_cache(cache_path, model=model, chunk_hashes=chunk_hashes)
        if cached is not None:
            vectors = cached
            norms = [_l2_norm(vector) for vector in vectors]
            return EmbeddingIndex(
                chunks=chunk_list,
                vectors=vectors,
                norms=norms,
                model=model,
                cache_path=str(cache_path),
                cache_hit=True,
            )

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(_embed_texts(batch, model=model, client=client))

    if len(vectors) != len(chunk_list):
        raise ValueError("Embedding response count did not match chunk count.")

    if cache_path is not None:
        _write_embedding_cache(cache_path, model=model, chunk_hashes=chunk_hashes, vectors=vectors)

    norms = [_l2_norm(vector) for vector in vectors]
    return EmbeddingIndex(
        chunks=chunk_list,
        vectors=vectors,
        norms=norms,
        model=model,
        cache_path=str(cache_path) if cache_path is not None else None,
        cache_hit=False,
    )


def build_retriever(
    chunks: Iterable[Chunk],
    *,
    backend: str = "bm25",
    **kwargs: Any,
) -> ChunkRetriever:
    """Build a retriever backend over chunks (bm25 by default)."""
    normalized = backend.lower().strip()
    if normalized == "bm25":
        k1 = float(kwargs.get("k1", 1.5))
        b = float(kwargs.get("b", 0.75))
        index = build_bm25_index(chunks, k1=k1, b=b)
        return BM25Retriever(index=index)
    if normalized in {"embeddings", "embedding", "openai-embeddings", "openai"}:
        model = str(kwargs.get("embedding_model") or kwargs.get("model") or "text-embedding-3-small")
        batch_size = int(kwargs.get("batch_size") or kwargs.get("embedding_batch_size") or 64)
        cache_dir = kwargs.get("cache_dir") or kwargs.get("embedding_cache_dir")
        client = kwargs.get("client")
        if client is not None and not isinstance(client, OpenAI):
            raise TypeError("client must be an OpenAI instance when provided.")
        index = build_embedding_index(
            chunks,
            model=model,
            batch_size=batch_size,
            client=client,
            cache_dir=cache_dir,
        )
        return EmbeddingRetriever(index=index, client=client, model=model)
    raise ValueError(f"Unsupported retrieval backend: {backend!r}")


def retrieve(query: str, index: ChunkIndex, *, top_k: int = 5) -> List[RetrievalHit]:
    """Return top-k chunks for a query using BM25."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    hits: List[RetrievalHit] = []
    for chunk, tf, doc_len in zip(index.chunks, index.term_freqs, index.doc_lens):
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            idf = index.idf.get(term, 0.0)
            denom = tf[term] + index.k1 * (1 - index.b + index.b * doc_len / index.avg_doc_len)
            score += idf * ((tf[term] * (index.k1 + 1)) / denom)
        if score > 0:
            hits.append(RetrievalHit(chunk=chunk, score=score))

    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[: max(top_k, 0)]


def build_chunk_index_from_pdf(
    pdf_path: str | Path,
    *,
    max_chunk_chars: int = 2000,
    use_ocr: bool = False,
    ocr_min_chars: int = 40,
    ocr_lang: str = "eng",
    ocr_dpi: int = 200,
) -> ChunkIndex:
    """Convenience helper to chunk and index a PDF."""
    chunks = chunk_pdf(
        pdf_path,
        max_chunk_chars=max_chunk_chars,
        use_ocr=use_ocr,
        ocr_min_chars=ocr_min_chars,
        ocr_lang=ocr_lang,
        ocr_dpi=ocr_dpi,
    )
    return build_bm25_index(chunks)


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 4:
        return False
    if _HEADING_PREFIX_RE.match(stripped):
        return True
    if _HEADING_NUMBERED_RE.match(stripped):
        return True
    if stripped.endswith(":") and len(stripped.split()) <= 8:
        return True
    if _ALL_CAPS_RE.match(stripped) and _count_alpha(stripped) >= 4:
        return True
    if _HEADING_KEYWORDS_RE.match(stripped):
        return True
    if "IN WITNESS WHEREOF" in stripped.upper():
        return True
    if _is_title_case_heading(stripped):
        return True
    return False


def _count_alpha(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _is_title_case_heading(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'/-]*", text)
    if not words or len(words) > 7:
        return False
    if text.endswith((".", ";")):
        return False
    capitalized = sum(1 for word in words if word[0].isupper())
    return (capitalized / len(words)) >= 0.8


def _split_long_chunk(
    *,
    page_num: int,
    section_index: int,
    heading: Optional[str],
    body: str,
    max_chunk_chars: int,
) -> List[Chunk]:
    if max_chunk_chars <= 0 or len(body) <= max_chunk_chars:
        return [
            Chunk(
                chunk_id=f"p{page_num}_c{section_index}",
                page_num=page_num,
                heading=heading,
                chunk_text=body,
            )
        ]

    parts = _split_text_by_chars(body, max_chunk_chars)
    chunks: List[Chunk] = []
    for part_index, part in enumerate(parts):
        suffix = f"_s{part_index}" if part_index > 0 else ""
        chunks.append(
            Chunk(
                chunk_id=f"p{page_num}_c{section_index}{suffix}",
                page_num=page_num,
                heading=heading,
                chunk_text=part,
            )
        )
    return chunks


def _split_text_by_chars(text: str, max_chars: int) -> List[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    words = text.split()
    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    for word in words:
        extra = len(word) + (1 if current else 0)
        if current_len + extra > max_chars and current:
            parts.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra

    if current:
        parts.append(" ".join(current))

    return parts


def _embed_texts(texts: List[str], *, model: str, client: OpenAI) -> List[List[float]]:
    response = client.embeddings.create(model=model, input=texts)
    data = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in data]


def _l2_norm(vector: List[float]) -> float:
    norm = math.sqrt(sum(value * value for value in vector))
    return norm if norm > 0 else 1.0


def _cosine_similarity(
    a: List[float],
    norm_a: float,
    b: List[float],
    norm_b: float,
) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError("Embedding vectors must have the same dimension.")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (norm_a * norm_b)


def _hash_chunk(chunk: Chunk) -> str:
    content = f"{chunk.chunk_id}|{chunk.page_num}|{chunk.heading or ''}|{chunk.chunk_text}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def _resolve_embedding_cache_path(
    *,
    cache_dir: Optional[str | Path],
    model: str,
    chunk_hashes: List[str],
    cache_key: Optional[str],
) -> Optional[Path]:
    if cache_dir is None:
        return None
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    if cache_key:
        key = cache_key
    else:
        joined = "".join(chunk_hashes)
        key = hashlib.sha256(joined.encode("utf-8")).hexdigest()

    safe_model = _safe_model_name(model)
    return cache_root / f"embeddings_{safe_model}_{key}.json"


def _load_embedding_cache(
    cache_path: Path,
    *,
    model: str,
    chunk_hashes: List[str],
) -> Optional[List[List[float]]]:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if data.get("model") != model:
        return None
    if data.get("chunk_hashes") != chunk_hashes:
        return None
    vectors = data.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(chunk_hashes):
        return None
    return vectors


def _write_embedding_cache(
    cache_path: Path,
    *,
    model: str,
    chunk_hashes: List[str],
    vectors: List[List[float]],
) -> None:
    payload = {
        "model": model,
        "chunk_hashes": chunk_hashes,
        "embeddings": vectors,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)
