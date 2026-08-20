"""
SQL query tests against the live program.db.
Discovers tables dynamically — does not hardcode dataset names.
Also tests the query_spreadsheet tool function directly.

Requires a completed ingest (python ingest.py) before running.
Run: pytest tests/test_sql.py -v
"""

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from tools import query_spreadsheet


@pytest.fixture(scope="module")
def db():
    con = sqlite3.connect(config.DB_PATH)
    yield con
    con.close()

@pytest.fixture(scope="module")
def tables(db):
    cur = db.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cur.fetchall()]

@pytest.fixture(scope="module")
def tables_with_domains(db, tables):
    """Subset of tables that have a domains column."""
    result = []
    cur = db.cursor()
    for table in tables:
        cur.execute(f"PRAGMA table_info({table})")
        cols = {row[1] for row in cur.fetchall()}
        if "domains" in cols:
            result.append(table)
    return result


# --- DB state ---

def test_db_has_tables(tables):
    assert tables, "No tables found — has ingest.py been run?"

def test_all_tables_have_rows(db, tables):
    cur = db.cursor()
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        assert count > 0, f"{table} is empty"


# --- Domain normalization (on any table that has a domains column) ---

def test_domains_uppercase_in_all_tables(db, tables_with_domains):
    if not tables_with_domains:
        pytest.skip("No tables with domains column")
    cur = db.cursor()
    for table in tables_with_domains:
        cur.execute(f"SELECT domains FROM {table} WHERE domains IS NOT NULL")
        for (domain,) in cur.fetchall():
            assert domain == domain.upper(), f"{table}: domain not uppercase: '{domain}'"
            assert not domain.startswith(","), f"{table}: leading comma: '{domain}'"
            assert not domain.endswith(","), f"{table}: trailing comma: '{domain}'"


# --- query_spreadsheet tool ---

def test_select_returns_results(tables):
    result = query_spreadsheet.invoke(f"SELECT * FROM {tables[0]} LIMIT 3")
    assert "error" not in result.lower()
    assert len(result.strip()) > 0

def test_non_select_rejected():
    result = query_spreadsheet.invoke("DROP TABLE gyf_dataset_literature")
    assert "only SELECT" in result

def test_bad_table_name_returns_sql_error():
    result = query_spreadsheet.invoke("SELECT * FROM nonexistent_table_xyz")
    assert "SQL error" in result or "error" in result.lower()

def test_domain_filter_query(tables_with_domains):
    if not tables_with_domains:
        pytest.skip("No tables with domains column")
    table = tables_with_domains[0]
    result = query_spreadsheet.invoke(
        f"SELECT COUNT(*) FROM {table} WHERE domains IS NOT NULL"
    )
    assert "error" not in result.lower()

def test_empty_query_returns_no_results_message(tables):
    result = query_spreadsheet.invoke(
        f"SELECT * FROM {tables[0]} WHERE 1=0"
    )
    assert "no results" in result.lower()
