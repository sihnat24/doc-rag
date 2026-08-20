"""
Embedding pipeline tests.
Validates that text chunks can be encoded into well-formed vectors —
independent of the persistent Chroma store.

Uses Nishizawa 1998 (short prose-only paper) rather than Singer to keep
this test focused on the encoder, not PDF parsing complexity.

Run: pytest tests/test_embeddings.py -v
"""

import os
import sys

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ingest import extract_chunks


@pytest.fixture(scope="module")
def encoder():
    return SentenceTransformer(config.ENCODER)


@pytest.fixture(scope="module")
def nishizawa_embeddings(nishizawa_parsed, encoder):
    prose, tables, figures, _ = nishizawa_parsed
    chunks = extract_chunks(prose) + tables + figures
    assert chunks, "No chunks produced — parsing may have failed"
    embeddings = encoder.encode(chunks)
    return chunks, embeddings


# --- Encoder sanity ---

def test_encoder_produces_output(nishizawa_embeddings):
    _, embeddings = nishizawa_embeddings
    assert embeddings is not None
    assert len(embeddings) > 0

def test_embedding_dimension(nishizawa_embeddings):
    """all-mpnet-base-v2 produces 768-dim vectors."""
    _, embeddings = nishizawa_embeddings
    assert embeddings.shape[1] == 768

def test_embedding_count_matches_chunks(nishizawa_embeddings):
    chunks, embeddings = nishizawa_embeddings
    assert embeddings.shape[0] == len(chunks)


# --- Vector quality ---

def test_no_zero_vectors(nishizawa_embeddings):
    _, embeddings = nishizawa_embeddings
    zero_rows = np.all(embeddings == 0, axis=1)
    assert not zero_rows.any(), f"{zero_rows.sum()} zero vectors — encoder may have failed silently"

def test_no_nan_values(nishizawa_embeddings):
    _, embeddings = nishizawa_embeddings
    assert not np.isnan(embeddings).any(), "NaN values in embeddings"

def test_distinct_chunks_produce_distinct_vectors(nishizawa_embeddings):
    chunks, embeddings = nishizawa_embeddings
    if len(chunks) < 2:
        pytest.skip("Not enough chunks to compare")
    assert not np.allclose(embeddings[0], embeddings[1]), \
        "First two chunks produced identical vectors — encoder may be returning constants"

def test_similar_chunks_are_closer_than_random(nishizawa_embeddings):
    """A chunk should be more similar to itself than to an unrelated chunk."""
    _, embeddings = nishizawa_embeddings
    if len(embeddings) < 10:
        pytest.skip("Not enough chunks")
    v0 = embeddings[0]
    self_sim = float(np.dot(v0, v0) / (np.linalg.norm(v0) ** 2))
    other_sim = float(np.dot(v0, embeddings[9]) / (np.linalg.norm(v0) * np.linalg.norm(embeddings[9])))
    assert self_sim > other_sim
