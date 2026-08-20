"""
Table ingestion tests.
Validates CSV and multi-sheet Excel ingestion into SQLite.
Uses GYF_dataset.xlsx — derives expected table names and row counts
from the actual file rather than hardcoding them.

Run: pytest tests/test_tables.py -v
"""

import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest import ingest_tables

GYF_PATH = Path("data/GYF_dataset.xlsx")


# --- Fixtures ---

@pytest.fixture(scope="module")
def gyf_sheets():
    """Read GYF Excel directly — ground truth for expected tables and row counts."""
    return pd.read_excel(GYF_PATH, sheet_name=None)


@pytest.fixture(scope="module")
def gyf_db(tmp_path_factory, gyf_sheets):
    """Ingest GYF_dataset.xlsx into a temp SQLite DB."""
    db_path = str(tmp_path_factory.mktemp("db") / "test.db")
    ingest_tables([GYF_PATH], log={}, force=True, db_path=db_path)
    con = sqlite3.connect(db_path)
    yield con
    con.close()


# --- Multi-sheet Excel: table creation ---

def test_all_sheets_become_tables(gyf_db, gyf_sheets):
    """Every sheet in the Excel should produce a SQLite table."""
    base = GYF_PATH.stem.lower().replace(" ", "_").replace("-", "_")
    expected = {
        f"{base}_{name.lower().replace(' ', '_')}"
        for name in gyf_sheets.keys()
    }
    cur = gyf_db.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    actual = {row[0] for row in cur.fetchall()}
    assert expected <= actual, f"Missing tables: {expected - actual}"

def test_no_extra_tables(gyf_db, gyf_sheets):
    """Ingesting one file should not create tables beyond its sheets."""
    base = GYF_PATH.stem.lower().replace(" ", "_").replace("-", "_")
    expected = {
        f"{base}_{name.lower().replace(' ', '_')}"
        for name in gyf_sheets.keys()
    }
    cur = gyf_db.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    actual = {row[0] for row in cur.fetchall()}
    assert actual == expected, f"Unexpected tables: {actual - expected}"


# --- Row counts derived from source file ---

def test_row_counts_match_source(gyf_db, gyf_sheets):
    base = GYF_PATH.stem.lower().replace(" ", "_").replace("-", "_")
    cur = gyf_db.cursor()
    for sheet_name, df in gyf_sheets.items():
        table = f"{base}_{sheet_name.lower().replace(' ', '_')}"
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        db_count = cur.fetchone()[0]
        assert db_count == len(df), \
            f"{table}: expected {len(df)} rows from source, got {db_count}"


# --- Data quality ---

def test_domains_uppercased(gyf_db, gyf_sheets):
    """Domains column should be fully uppercase with no trailing commas."""
    base = GYF_PATH.stem.lower().replace(" ", "_").replace("-", "_")
    cur = gyf_db.cursor()
    for sheet_name in gyf_sheets.keys():
        table = f"{base}_{sheet_name.lower().replace(' ', '_')}"
        cur.execute(f"PRAGMA table_info({table})")
        cols = {row[1] for row in cur.fetchall()}
        if "domains" not in cols:
            continue
        cur.execute(f"SELECT domains FROM {table} WHERE domains IS NOT NULL")
        for (domain,) in cur.fetchall():
            assert domain == domain.upper(), f"Domain not uppercase: '{domain}'"
            assert not domain.endswith(","), f"Trailing comma: '{domain}'"
            assert not domain.startswith(","), f"Leading comma: '{domain}'"

def test_column_names_normalized(gyf_db, gyf_sheets):
    """Column names should be lowercase with underscores, no leading/trailing spaces."""
    base = GYF_PATH.stem.lower().replace(" ", "_").replace("-", "_")
    cur = gyf_db.cursor()
    for sheet_name in gyf_sheets.keys():
        table = f"{base}_{sheet_name.lower().replace(' ', '_')}"
        cur.execute(f"PRAGMA table_info({table})")
        for row in cur.fetchall():
            col = row[1]
            assert col == col.strip(), f"Column has surrounding whitespace: '{col}'"
            assert col == col.lower(), f"Column not lowercased: '{col}'"


# --- CSV ingestion (synthetic fixture) ---

def test_csv_ingestion(tmp_path):
    """A single CSV should create one table named after the file."""
    csv_path = tmp_path / "test_data.csv"
    csv_path.write_text("name,value,score\nalpha,1,0.9\nbeta,2,0.7\n")

    db_path = str(tmp_path / "csv_test.db")
    ingest_tables([csv_path], log={}, force=True, db_path=db_path)

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert "test_data" in tables

    cur.execute("SELECT COUNT(*) FROM test_data")
    assert cur.fetchone()[0] == 2
    con.close()
