"""In-memory, upload-only PDF extraction with page-preserving chunks."""

from __future__ import annotations

import re
import uuid

import fitz

from financial_analyst.models import DocumentChunk, UploadedDocument


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
                )
            )
            index += 1
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks
