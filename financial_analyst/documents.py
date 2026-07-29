"""In-memory, upload-only PDF extraction with page-preserving chunks."""

from __future__ import annotations

import re
import uuid
from collections import Counter
from collections.abc import Iterable
from math import log, sqrt

import fitz

from financial_analyst.models import (
    Claim,
    ConfidenceCategory,
    DocumentChunk,
    EvidenceRef,
    RetrievalHit,
    SupportStatus,
    UploadedDocument,
)

_INSTRUCTION_PATTERN = re.compile(
    r"(?i)\b(?:ignore|disregard|override|reveal|print|expose|system prompt|api key|secret)\b"
)
_SEMANTIC_CONCEPTS = {
    "revenue": {"revenue", "sales", "turnover", "topline", "income"},
    "profit": {"profit", "earnings", "income", "margin", "profitable"},
    "cash": {"cash", "liquidity", "funds", "reserves"},
    "debt": {"debt", "borrowings", "leverage", "loans", "liabilities"},
    "investment": {"capex", "capital", "expenditure", "investment", "spending"},
    "risk": {"risk", "uncertainty", "exposure", "threat", "challenge"},
    "guidance": {"guidance", "outlook", "forecast", "expectation", "projection"},
    "customer": {"customer", "client", "subscriber", "buyer"},
    "competition": {"competition", "competitor", "rival", "marketshare"},
    "regulation": {"regulation", "regulatory", "compliance", "legal"},
}


class DocumentValidationError(ValueError):
    """Raised when an uploaded document cannot be processed safely."""


def sanitize_filename(filename: str) -> str:
    """Remove directories and unsupported characters from an uploaded name."""

    basename = filename.replace("\\", "/").split("/")[-1].strip()
    sanitized = re.sub(r"[^A-Za-z0-9._ -]", "_", basename)
    sanitized = re.sub(r"\s+", " ", sanitized).strip(" .")
    return (sanitized or "uploaded.pdf")[:120]


def extract_pdf_upload(
    filename: str,
    content: bytes,
    *,
    max_size_mb: int,
    chunk_size: int = 1800,
    chunk_overlap: int = 200,
) -> UploadedDocument:
    """Extract every page directly from uploaded bytes; no arbitrary path is accepted."""

    if len(content) > max_size_mb * 1024 * 1024:
        raise DocumentValidationError(f"PDF exceeds the {max_size_mb} MB upload limit.")
    if not content.startswith(b"%PDF"):
        raise DocumentValidationError("The uploaded file is not a valid PDF.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")

    safe_filename = sanitize_filename(filename)
    if not safe_filename.lower().endswith(".pdf"):
        safe_filename = f"{safe_filename}.pdf"
    document_id = uuid.uuid4().hex
    chunks: list[DocumentChunk] = []
    extraction_errors: list[str] = []

    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            for page_index in range(document.page_count):
                page_number = page_index + 1
                try:
                    text = document.load_page(page_index).get_text("text").strip()
                except (RuntimeError, ValueError) as error:
                    extraction_errors.append(f"Page {page_number}: {type(error).__name__}")
                    continue
                if not text:
                    extraction_errors.append(f"Page {page_number}: no extractable text")
                    continue
                chunks.extend(
                    _chunk_page(
                        document_id=document_id,
                        safe_filename=safe_filename,
                        page_number=page_number,
                        text=text,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
            page_count = document.page_count
    except (fitz.FileDataError, RuntimeError, ValueError) as error:
        raise DocumentValidationError(f"PDF extraction failed: {type(error).__name__}") from error

    if not chunks:
        raise DocumentValidationError(
            "The PDF contained no extractable text. Scanned PDFs require OCR, which is not enabled."
        )
    return UploadedDocument(
        document_id=document_id,
        safe_filename=safe_filename,
        page_count=page_count,
        chunks=chunks,
        extraction_errors=extraction_errors,
    )


def _chunk_page(
    *,
    document_id: str,
    safe_filename: str,
    page_number: int,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            paragraph_break = text.rfind("\n", start, end)
            if paragraph_break > start + chunk_size // 2:
                end = paragraph_break
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    safe_filename=safe_filename,
                    page_number=page_number,
                    chunk_index=index,
                    text=chunk_text,
                    character_start=start,
                    character_end=end,
                    contains_instruction_like_text=bool(_INSTRUCTION_PATTERN.search(chunk_text)),
                    local_embedding=_semantic_vector(_terms(chunk_text)),
                )
            )
            index += 1
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def hybrid_retrieve(
    documents: Iterable[UploadedDocument],
    query: str,
    *,
    limit: int = 8,
    minimum_score: float = 0.08,
    lexical_weight: float = 0.55,
) -> list[RetrievalHit]:
    """Return page-aware lexical and local semantic evidence above a threshold.

    The compact semantic representation maps finance paraphrases into shared
    vector dimensions. It runs locally without a paid API, model download, or
    persistent vector store, and lexical ranking remains available by design.
    """

    chunks = [chunk for document in documents for chunk in document.chunks]
    if not chunks or not query.strip() or limit <= 0:
        return []
    query_terms = _terms(query)
    if not query_terms:
        return []
    document_terms = [_terms(chunk.text) for chunk in chunks]
    document_frequency = Counter(term for terms in document_terms for term in set(terms))
    lexical_raw = [
        _bm25_score(query_terms, terms, document_frequency, len(chunks)) for terms in document_terms
    ]
    lexical_max = max(lexical_raw, default=0.0)
    query_vector = _semantic_vector(query_terms)
    semantic_raw = [
        _cosine(query_vector, chunk.local_embedding or _semantic_vector(terms))
        for chunk, terms in zip(chunks, document_terms, strict=True)
    ]

    candidates: list[RetrievalHit] = []
    for chunk, lexical, semantic in zip(chunks, lexical_raw, semantic_raw, strict=True):
        normalized_lexical = lexical / lexical_max if lexical_max > 0 else 0.0
        hybrid = lexical_weight * normalized_lexical + (1 - lexical_weight) * semantic
        if hybrid < minimum_score:
            continue
        candidates.append(
            RetrievalHit(
                chunk_id=f"{chunk.document_id}-p{chunk.page_number}-c{chunk.chunk_index}",
                document_id=chunk.document_id,
                safe_filename=chunk.safe_filename,
                page_number=chunk.page_number,
                text=chunk.text,
                lexical_score=round(normalized_lexical, 6),
                semantic_score=round(semantic, 6),
                hybrid_score=round(hybrid, 6),
                retrieval_method="local_concept_embedding+bm25",
                character_start=chunk.character_start,
                character_end=chunk.character_end,
                contains_instruction_like_text=chunk.contains_instruction_like_text,
            )
        )
    ranked = sorted(
        candidates,
        key=lambda hit: (-hit.hybrid_score, hit.page_number, hit.chunk_id),
    )
    return _deduplicate_hits(ranked, limit=limit)


def semantic_similarity(left: str, right: str) -> float:
    """Return local concept-embedding cosine similarity for two text passages."""

    return _cosine(_semantic_vector(_terms(left)), _semantic_vector(_terms(right)))


def verify_qualitative_claims(
    synthesis: str,
    evidence: list[EvidenceRef],
) -> tuple[str, list[Claim]]:
    """Remove assertive qualitative wording that lacks semantic evidence support."""

    retained: list[str] = []
    claims: list[Claim] = []
    removed = False
    evidence_text = {
        item.evidence_id: " ".join(
            part
            for part in (item.title, item.excerpt, item.metric, item.source, item.definition)
            if part
        )
        for item in evidence
    }
    for index, line in enumerate(synthesis.splitlines(), start=1):
        stripped = line.strip().lstrip("- ").strip()
        if not stripped or line.lstrip().startswith("#") or line.startswith("_"):
            retained.append(line)
            continue
        scores = sorted(
            (
                (semantic_similarity(stripped, text), evidence_id)
                for evidence_id, text in evidence_text.items()
            ),
            reverse=True,
        )
        best_score, best_id = scores[0] if scores else (0.0, "")
        generic_caveat = bool(
            re.search(
                r"(?i)\b(?:missing|unavailable|uncertain|evidence|source|provider|"
                r"assumption|risk|limitation|conflict|partial)\b",
                stripped,
            )
        )
        causal = bool(
            re.search(r"(?i)\b(?:caused|causes|because|driven by|resulted in|led to)\b", stripped)
        )
        supported = best_score >= (0.22 if causal else 0.10)
        if supported:
            status = SupportStatus.VERIFIED
            category = "qualitative_interpretation"
            reason = f"Local semantic evidence similarity {best_score:.3f}."
            retained.append(line)
        elif generic_caveat:
            status = SupportStatus.NOT_VERIFIABLE
            category = "qualitative_caveat"
            reason = "Conservative limitation wording does not assert a new external fact."
            retained.append(line)
        else:
            status = SupportStatus.UNSUPPORTED
            category = "removed_interpretation"
            reason = "No supplied evidence passage semantically supports this wording."
            removed = True
        claims.append(
            Claim(
                claim_id=f"claim-qualitative-{index}",
                text=stripped,
                category=category,
                claim_type="interpretive",
                evidence_ids=[best_id] if supported and best_id else [],
                support_status=status,
                confidence_category=(
                    ConfidenceCategory.MODERATE if supported else ConfidenceCategory.INSUFFICIENT
                ),
                verification_reason=reason,
            )
        )
    if removed:
        retained.append(
            "\n_Unsupported qualitative wording was omitted by the semantic evidence gate._"
        )
    return "\n".join(retained).strip(), claims


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _bm25_score(
    query: list[str],
    document: list[str],
    document_frequency: Counter[str],
    document_count: int,
) -> float:
    if not document:
        return 0.0
    frequencies = Counter(document)
    score = 0.0
    for term in set(query):
        frequency = frequencies[term]
        if not frequency:
            continue
        inverse_frequency = log(
            1 + (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
        )
        score += inverse_frequency * (frequency * 2.2) / (frequency + 1.2)
    return score


def _semantic_vector(terms: list[str]) -> dict[str, float]:
    vector: Counter[str] = Counter()
    for term in terms:
        matched = False
        for concept, vocabulary in _SEMANTIC_CONCEPTS.items():
            if term in vocabulary:
                vector[f"concept:{concept}"] += 1.0
                matched = True
        if not matched and len(term) >= 4:
            vector[f"token:{term}"] += 0.25
    norm = sqrt(sum(value * value for value in vector.values()))
    return {key: value / norm for key, value in vector.items()} if norm else {}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    return max(0.0, sum(value * right.get(key, 0.0) for key, value in left.items()))


def _deduplicate_hits(hits: list[RetrievalHit], *, limit: int) -> list[RetrievalHit]:
    output: list[RetrievalHit] = []
    normalized_seen: set[str] = set()
    for hit in hits:
        normalized = " ".join(_terms(hit.text))
        if normalized in normalized_seen:
            continue
        if any(
            existing.document_id == hit.document_id
            and existing.page_number == hit.page_number
            and _overlap(existing, hit) >= 0.75
            for existing in output
        ):
            continue
        normalized_seen.add(normalized)
        output.append(hit)
        if len(output) >= limit:
            break
    return output


def _overlap(left: RetrievalHit, right: RetrievalHit) -> float:
    intersection = max(
        0,
        min(left.character_end, right.character_end)
        - max(left.character_start, right.character_start),
    )
    shortest = max(
        1,
        min(
            left.character_end - left.character_start,
            right.character_end - right.character_start,
        ),
    )
    return intersection / shortest
