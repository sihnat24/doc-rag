import sqlite3

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
import config
import ollama

encoder = SentenceTransformer(config.ENCODER)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(config.COLLECTION)

_all_data = collection.get(include=["documents", "metadatas"])
_all_docs = _all_data["documents"]
_all_metas = _all_data["metadatas"]
# TODO: whitespace split means punctuation attached to words ("Budget." != "Budget") and
# hyphenated terms ("risk-status" != "risk") are treated as different tokens, hurting recall.
# Production would use NLTK/spaCy tokenization with punctuation stripping.
_bm25 = BM25Okapi([doc.lower().split() for doc in _all_docs])

def _hybrid_search(query: str, n_results: int):
    RRF_K = 60

    # Dense retrieval
    q_emb = encoder.encode(query).tolist()
    dense_results = collection.query(query_embeddings=[q_emb], n_results=n_results * 2)
    dense_docs = dense_results["documents"][0]
    dense_sources = [m["source"] for m in dense_results["metadatas"][0]]

    # BM25 retrieval
    tokens = query.lower().split()
    bm25_scores = _bm25.get_scores(tokens)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n_results * 2]

    # RRF merge: each doc accumulates 1/(k + rank) from each retriever
    rrf_scores = {}
    doc_source_map = {}

    for rank, (doc, source) in enumerate(zip(dense_docs, dense_sources)):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (RRF_K + rank + 1)
        doc_source_map[doc] = source

    for rank, idx in enumerate(top_bm25_idx):
        doc = _all_docs[idx]
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (RRF_K + rank + 1)
        doc_source_map[doc] = _all_metas[idx]["source"]

    sorted_docs = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[:n_results]
    return sorted_docs, [doc_source_map[d] for d in sorted_docs]


@tool
def search_knowledge_base(query: str) -> str:
    """Search the local knowledge base for information relevant to the query."""

    chunks, sources = _hybrid_search(query, config.TOP_K)

    output = []
    for count, (chunk, source) in enumerate(zip(chunks, sources), start=1):
        output.append(f"source {count}: {source}, {chunk}.")

    return " ".join(output)

@tool
def list_knowledge_base() -> str:
    """List all documents and tables currently indexed in the local knowledge base."""
    results = collection.get(include=["metadatas"])
    sources = {m["source"] for m in results["metadatas"] if "source" in m}

    con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    con.close()

    output = []
    if sources:
        output.append("Documents (vector search):\n" + "\n".join(f"- {s}" for s in sorted(sources)))
    if tables:
        output.append("Tables (SQL query):\n" + "\n".join(f"- {t}" for t in sorted(tables)))
    return "\n\n".join(output) if output else "No documents or tables found in the knowledge base."


web_search = DuckDuckGoSearchRun()


@tool
def query_spreadsheet(sql: str) -> str:
    """Run a read-only SQL SELECT query against the paper metadata database.

    Table: example_domain_data
    Columns: pmid, publication, year, first_author, domains,
             experimental_methods, "data_extracted?"

    Only SELECT statements are permitted. Use this for precise lookups,
    filtering, or aggregations (e.g. all papers using NMR, papers by a specific author, papers in a domain).
    """
    sql = sql.strip()
    if not sql.upper().startswith("SELECT"):
        return "Error: only SELECT queries are permitted."

    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        col_names = [d[0] for d in cur.description]
        con.close()
    except sqlite3.OperationalError as e:
        return f"SQL error: {e}"

    if not rows:
        return "Query returned no results."

    header = " | ".join(col_names)
    divider = "-" * len(header)
    lines = [header, divider] + [" | ".join(str(v) for v in row) for row in rows]
    return "\n".join(lines)


def get_answer(question, chunks, sources):

    prompt = "\n---\n".join(f"[{src}]\n{chunk}" for src, chunk in    
  zip(sources, chunks))  
    prompt += f"\n\nQuestion: {question}"

    response = ollama.chat(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": config.ANSWER_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    answer = response["message"]["content"]
    return answer





