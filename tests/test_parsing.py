"""
Parsing validation tests.
Verifies that text extraction works correctly for different file formats
and layout types (1-column, 2-column, mixed layout PDFs).

Run: pytest tests/test_parsing.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest import parse_pdf, parse_html, parse_csv, detect_columns, extract_chunks
import pdfplumber


# --- PDF: Singer 2024 (mixed layout — 1-col abstract + 2-col body with tables/figures) ---
# Known limitation: affiliation sidebars and funding blocks bleed into body text
# because the narrow sidebar columns are not cleanly separated by column detection.

@pytest.fixture(scope="module")
def singer_text():
    text = parse_pdf("data/singer_2024_38989636.pdf")
    chunks = extract_chunks(text)
    with open("tests/singer_chunks.txt", "w") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"--- CHUNK {i} ---\n{chunk}\n\n")
    return text

def test_singer_text_non_empty(singer_text):
    assert len(singer_text.strip()) > 1000

def test_singer_abstract_present(singer_text):
    """Core abstract sentence should survive extraction."""
    assert "Short sequences that mediate interactions with modular binding domains" in singer_text

def test_singer_keywords_present(singer_text):
    assert "EVH1" in singer_text

def test_singer_body_content_present(singer_text):
    """Key scientific claim from 2-col body should be present."""
    assert "incomplete divergence" in singer_text

def test_singer_author_present(singer_text):
    assert "Singer" in singer_text


# --- PDF: Column detection ---

def test_column_detection_two_column():
    """A typical 2-col academic paper page should detect 2 columns."""
    with pdfplumber.open("data/singer_2024_38989636.pdf") as pdf:
        # Page 2+ are typically 2-column body
        for page in pdf.pages[1:4]:
            cols = detect_columns(page, pixel_thresh=15)
            # At least some body pages should be detected as 2-col
            if cols == 2:
                return
    pytest.fail("No 2-column pages detected in singer_2024 — column detection may be broken")

def test_column_detection_first_page():
    """First page of Singer 2024 has a mixed layout — should not crash."""
    with pdfplumber.open("data/singer_2024_38989636.pdf") as pdf:
        page = pdf.pages[0]
        cols = detect_columns(page, pixel_thresh=15)
        assert cols in (1, 2)


# --- PDF: Older 2-col paper (Kofler 2005) ---

@pytest.fixture(scope="module")
def kofler_text():
    return parse_pdf("data/kofler_2005_16120600.pdf")

def test_kofler_text_non_empty(kofler_text):
    assert len(kofler_text.strip()) > 1000

def test_kofler_known_content(kofler_text):
    """GYF domain content should be present."""
    assert "GYF" in kofler_text

def test_kofler_reasonable_length(kofler_text):
    """Multi-page paper should produce substantial text."""
    assert len(kofler_text) > 5000


# --- PDF: Short paper (Golemi-Kotra 2003 — only 26 chunks, shortest in corpus) ---

@pytest.fixture(scope="module")
def golemi_text():
    return parse_pdf("data/golemi-kotra_2003_14709031.pdf")

def test_golemi_non_empty(golemi_text):
    assert len(golemi_text.strip()) > 500

def test_golemi_known_content(golemi_text):
    assert "EVH1" in golemi_text or "Mena" in golemi_text
