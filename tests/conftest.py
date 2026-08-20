"""
Session-scoped fixtures shared across test modules.
Marker converter is expensive to load — initialised once per test session.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest import extract_chunks, parse_pdf_marker
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict


@pytest.fixture(scope="session")
def converter():
    return PdfConverter(artifact_dict=create_model_dict())


@pytest.fixture(scope="session")
def singer_parsed(converter):
    """Singer 2024 — complex layout, tables, figures. Used by test_parsing."""
    prose, tables, figures = parse_pdf_marker(
        "data/singer_2024_38989636.pdf", converter
    )
    all_chunks = extract_chunks(prose) + tables + figures

    with open("tests/singer_chunks.txt", "w") as f:
        prose_chunks = extract_chunks(prose)
        f.write(f"=== PROSE CHUNKS ({len(prose_chunks)}) ===\n\n")
        for i, c in enumerate(prose_chunks, 1):
            f.write(f"--- PROSE {i} ---\n{c}\n\n")
        f.write(f"=== TABLE CHUNKS ({len(tables)}) ===\n\n")
        for i, c in enumerate(tables, 1):
            f.write(f"--- TABLE {i} ---\n{c}\n\n")
        f.write(f"=== FIGURE CHUNKS ({len(figures)}) ===\n\n")
        for i, c in enumerate(figures, 1):
            f.write(f"--- FIGURE {i} ---\n{c}\n\n")

    return prose, tables, figures, all_chunks


@pytest.fixture(scope="session")
def nishizawa_parsed(converter):
    """Nishizawa 1998 — short, prose-only paper. Used by test_embeddings."""
    prose, tables, figures = parse_pdf_marker(
        "data/nishizawa_1998_9843987.pdf", converter
    )
    all_chunks = extract_chunks(prose) + tables + figures
    return prose, tables, figures, all_chunks
