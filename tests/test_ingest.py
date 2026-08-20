"""
Ingestion validation tests.
Verifies that the SQLite tables and Chroma vector store reflect
the expected state after a clean ingest of data/.
Run after ingest.py: pytest tests/test_ingest.py -v
"""

import sqlite3
import pytest
import chromadb
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ingest import extract_chunks, file_hash
from pathlib import Path


# --- Fixtures ---

@pytest.fixture(scope="module")
def db():
    con = sqlite3.connect(config.DB_PATH)
    yield con
    con.close()

@pytest.fixture(scope="module")
def chroma():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection(config.COLLECTION)


# --- SQLite: table existence ---

EXPECTED_TABLES = {
    "wh1_dataset_literature",
    "wh1_dataset_data",
    "gyf_dataset_literature",
    "gyf_dataset_data",
}

def test_expected_tables_exist(db):
    cur = db.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert EXPECTED_TABLES <= tables, f"Missing tables: {EXPECTED_TABLES - tables}"

def test_no_unexpected_tables(db):
    cur = db.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    unexpected = tables - EXPECTED_TABLES
    assert not unexpected, f"Unexpected tables found: {unexpected}"


# --- SQLite: row counts ---

@pytest.mark.parametrize("table,expected_rows", [
    ("wh1_dataset_literature", 47),
    ("wh1_dataset_data", 770),
    ("gyf_dataset_literature", 14),
    ("gyf_dataset_data", 173),
])
def test_table_row_counts(db, table, expected_rows):
    cur = db.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    assert count == expected_rows, f"{table}: expected {expected_rows} rows, got {count}"


# --- SQLite: data quality ---

@pytest.mark.parametrize("table", ["gyf_dataset_literature", "wh1_dataset_literature"])
def test_domains_normalized(db, table):
    """Domains should be uppercase and have no trailing commas."""
    cur = db.cursor()
    cur.execute(f"SELECT domains FROM {table} WHERE domains IS NOT NULL")
    for (domain,) in cur.fetchall():
        assert domain == domain.upper(), f"Domain not uppercase: '{domain}'"
        assert not domain.endswith(","), f"Domain has trailing comma: '{domain}'"
        assert not domain.startswith(","), f"Domain has leading comma: '{domain}'"


# --- Chroma: document count ---

EXPECTED_PDF_SOURCES = {
    "ball_2000_10990454.pdf", "barzik_2001_11491285.pdf", "beneken_2000_10798399.pdf",
    "bertazon_2024_39308865.pdf", "boeda_2007_18158903.pdf", "carl_1999_10498433.pdf",
    "freund_1999_10404223.pdf", "freund_2002_12426371.pdf", "gertler_1996_8861907.pdf",
    "golemi-kotra_2003_14709031.pdf", "gu_2005_15850374.pdf", "hale_2012_22209900.pdf",
    "holtzman_2007_17973491.pdf", "kofler_2004_15105431.pdf", "kofler_2005_16000308.pdf",
    "kofler_2005_16120600.pdf", "kofler_2006_16403013.pdf", "kofler_2009_19483244.pdf",
    "nishizawa_1998_9843987.pdf", "peterson_2009_19273103.pdf", "prehoda_1999_10338211.pdf",
    "singer_2024_38989636.pdf", "ueki_2019_31585692.pdf", "uryga-polowy_2008_18803191.pdf",
    "volkman_2002_12437929.pdf", "zettl_2002_12372256.pdf", "zimmerman_2003_12857736.pdf",
}

def test_all_pdfs_indexed(chroma):
    results = chroma.get(include=["metadatas"])
    indexed_sources = {m["source"] for m in results["metadatas"]}
    missing = EXPECTED_PDF_SOURCES - indexed_sources
    assert not missing, f"PDFs not indexed: {missing}"

def test_no_extra_sources(chroma):
    results = chroma.get(include=["metadatas"])
    indexed_sources = {m["source"] for m in results["metadatas"]}
    extra = indexed_sources - EXPECTED_PDF_SOURCES
    assert not extra, f"Unexpected sources in index: {extra}"


# --- Chroma: chunk quality ---

def test_no_empty_chunks(chroma):
    results = chroma.get(include=["documents"])
    empty = [d for d in results["documents"] if not d or not d.strip()]
    assert not empty, f"Found {len(empty)} empty chunks"

def test_chunk_sizes_within_bounds(chroma):
    results = chroma.get(include=["documents"])
    oversized = [d for d in results["documents"] if len(d) > config.CHUNK_SIZE * 2]
    assert not oversized, f"Found {len(oversized)} chunks exceeding 2x CHUNK_SIZE"

def test_all_chunks_have_source_metadata(chroma):
    results = chroma.get(include=["metadatas"])
    missing = [m for m in results["metadatas"] if "source" not in m]
    assert not missing, f"Found {len(missing)} chunks missing source metadata"


# --- Unit: chunking function ---

def test_extract_chunks_respects_size():
    text = "word " * 1000
    chunks = extract_chunks(text)
    for chunk in chunks:
        assert len(chunk) <= config.CHUNK_SIZE * 2

def test_extract_chunks_no_empty():
    text = "This is a test document with enough content to chunk properly. " * 20
    chunks = extract_chunks(text)
    assert all(c.strip() for c in chunks)

def test_extract_chunks_covers_full_text():
    text = "alpha " * 500
    chunks = extract_chunks(text)
    assert len(chunks) > 1


# --- Unit: hash function ---

def test_file_hash_deterministic(tmp_path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello world")
    assert file_hash(f) == file_hash(f)

def test_file_hash_changes_on_content_change(tmp_path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"version 1")
    h1 = file_hash(f)
    f.write_bytes(b"version 2")
    h2 = file_hash(f)
    assert h1 != h2
