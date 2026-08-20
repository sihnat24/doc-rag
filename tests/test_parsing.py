"""
Parsing validation tests for Marker-based PDF ingestion.
Verifies prose/table/figure extraction on a representative paper.

Run: pytest tests/test_parsing.py -v
Dumps chunks to tests/singer_chunks.txt for manual inspection.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest import parse_pdf_marker, extract_chunks

# converter and singer_parsed come from conftest.py


def test_singer_prose_non_empty(singer_parsed):
    prose, *_ = singer_parsed
    assert len(prose.strip()) > 1000

def test_singer_abstract_present(singer_parsed):
    prose, *_ = singer_parsed
    assert "Short sequences that mediate interactions with modular binding domains" in prose

def test_singer_keywords_present(singer_parsed):
    prose, *_ = singer_parsed
    assert "EVH1" in prose

def test_singer_body_content_present(singer_parsed):
    prose, *_ = singer_parsed
    assert "incomplete divergence" in prose

def test_singer_has_tables(singer_parsed):
    _, tables, _, _ = singer_parsed
    assert len(tables) > 0, "Expected at least one table chunk"

def test_singer_table_chunks_have_pipe_rows(singer_parsed):
    _, tables, _, _ = singer_parsed
    for t in tables:
        assert "|" in t, f"Table chunk missing markdown rows:\n{t}"

def test_singer_has_figures(singer_parsed):
    _, _, figures, _ = singer_parsed
    assert len(figures) > 0, "Expected at least one figure chunk"

def test_singer_no_image_refs_in_chunks(singer_parsed):
    *_, all_chunks = singer_parsed
    image_refs = [c for c in all_chunks if c.strip().startswith("![")]
    assert not image_refs, f"Found {len(image_refs)} raw image ref chunks"

def test_singer_no_empty_chunks(singer_parsed):
    *_, all_chunks = singer_parsed
    empty = [c for c in all_chunks if not c.strip()]
    assert not empty, f"Found {len(empty)} empty chunks"

def test_singer_prose_chunks_within_bounds(singer_parsed):
    import config
    prose, tables, figures, _ = singer_parsed
    prose_chunks = extract_chunks(prose)
    oversized = [c for c in prose_chunks if len(c) > config.CHUNK_SIZE * 2]
    assert not oversized, f"{len(oversized)} prose chunks exceed 2x CHUNK_SIZE"

def test_singer_table_chunks_within_embedding_limit(singer_parsed):
    # Tables bypass the splitter — check they won't be silently truncated by the encoder.
    # all-mpnet-base-v2 max is 384 tokens (~1500 chars). Flag anything 2x over CHUNK_SIZE
    # as a sign a table is too large to embed meaningfully.
    import config
    _, tables, _, _ = singer_parsed
    oversized = [t for t in tables if len(t) > config.CHUNK_SIZE * 2]
    assert not oversized, (
        f"{len(oversized)} table chunks likely exceed embedding token limit: "
        + str([len(t) for t in oversized])
    )


# --- Kofler 2005: older 2-col paper, GYF domain ---

@pytest.fixture(scope="module")
def kofler_parsed(converter):
    prose, tables, figures = parse_pdf_marker(
        "data/kofler_2005_16120600.pdf", converter
    )
    return prose, tables, figures

def test_kofler_prose_non_empty(kofler_parsed):
    prose, *_ = kofler_parsed
    assert len(prose.strip()) > 1000

def test_kofler_known_content(kofler_parsed):
    prose, *_ = kofler_parsed
    assert "GYF" in prose

def test_kofler_reasonable_length(kofler_parsed):
    prose, *_ = kofler_parsed
    assert len(prose) > 5000


# --- Golemi-Kotra 2003: shortest paper in corpus ---

@pytest.fixture(scope="module")
def golemi_parsed(converter):
    prose, tables, figures = parse_pdf_marker(
        "data/golemi-kotra_2003_14709031.pdf", converter
    )
    return prose, tables, figures

def test_golemi_non_empty(golemi_parsed):
    prose, *_ = golemi_parsed
    assert len(prose.strip()) > 500

def test_golemi_known_content(golemi_parsed):
    prose, *_ = golemi_parsed
    assert "EVH1" in prose or "Mena" in prose
