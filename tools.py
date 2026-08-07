import sqlite3

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from sentence_transformers import SentenceTransformer
import chromadb
import config
import ollama

encoder = SentenceTransformer(config.ENCODER)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(config.COLLECTION)

@tool
def search_knowledge_base(query: str) -> str:
    """Search the local knowledge base for information relevant to the query."""

    q_emb = encoder.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=config.TOP_K)

    chunks = results["documents"][0]
    sources = [m['source'] for m in results["metadatas"][0]]

    output = []
    for count, (chunk, source) in enumerate(zip(chunks, sources), start=1):
        output.append(f"source {count}: {source}, {chunk}.")

    return " ".join(output)

@tool
def list_knowledge_base() -> str:
    """List all documents currently indexed in the local knowledge base."""
    results = collection.get(include=["metadatas"])
    sources = {m["source"] for m in results["metadatas"] if "source" in m}
    if not sources:
        return "No documents found in the knowledge base."
    return "Documents in knowledge base:\n" + "\n".join(f"- {s}" for s in sorted(sources))


web_search = DuckDuckGoSearchRun()


@tool
def query_spreadsheet(sql: str) -> str:
    """Run a read-only SQL SELECT query against the program risk/budget database.

    Table: risk_budget_tracker
    Columns: subsystem, vendor_under_eval, quarter, risk_status,
             risk_description, allocated_budget_usd, expenditure_usd,
             program_manager_note

    Only SELECT statements are permitted. Use this for precise lookups,
    filtering, or aggregations (e.g. total spend by vendor, all Red-risk rows).
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





