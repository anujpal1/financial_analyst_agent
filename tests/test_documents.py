from __future__ import annotations

import fitz
import pytest

from financial_analyst.documents import (
    DocumentValidationError,
    extract_pdf_upload,
    sanitize_filename,
)


def _pdf_bytes(page_count: int = 3) -> bytes:
    document = fitz.open()
    for page_number in range(1, page_count + 1):
        page = document.new_page()
        page.insert_text((72, 72), f"Revenue evidence from page {page_number}.")
    content = document.tobytes()
    document.close()
    return content


def test_pdf_filename_traversal_is_removed() -> None:
    assert sanitize_filename("../../private/statement.pdf") == "statement.pdf"
    assert sanitize_filename(r"..\..\private\statement.pdf") == "statement.pdf"


def test_uploaded_document_preserves_all_page_numbers() -> None:
    uploaded = extract_pdf_upload(
        "../../statement.pdf",
        _pdf_bytes(page_count=7),
        max_size_mb=2,
    )
    assert uploaded.safe_filename == "statement.pdf"
    assert uploaded.page_count == 7
    assert {chunk.page_number for chunk in uploaded.chunks} == set(range(1, 8))
    assert all(chunk.local_embedding for chunk in uploaded.chunks)
    assert "local_embedding" not in uploaded.model_dump()["chunks"][0]


def test_non_pdf_is_rejected() -> None:
    with pytest.raises(DocumentValidationError, match="valid PDF"):
        extract_pdf_upload("not.pdf", b"not a pdf", max_size_mb=1)


def test_upload_size_limit_is_enforced() -> None:
    with pytest.raises(DocumentValidationError, match="upload limit"):
        extract_pdf_upload("large.pdf", b"%PDF" + b"x" * 1024, max_size_mb=0)
