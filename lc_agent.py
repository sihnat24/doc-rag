# lc_agent.py — LangChain-based agent with re-ranking and structured CSV tool
#
# Install: pip install langchain langchain-ollama langchain-chroma
#
# Key LangChain concepts used here:
#   @tool               — decorator that turns a Python function into a LangChain tool.
#                         The docstring becomes the tool description the model reads.
#   create_tool_calling_agent — builds an agent that uses tool-calling (modern API, not ReAct).
#   AgentExecutor       — runs the agentic loop: calls tool → feeds result back → repeats
#                         until model stops calling tools.
#   {agent_scratchpad}  — placeholder in the prompt where LangChain injects tool call history.
#   verbose=True        — prints each step of the loop so you can see the agent reasoning.


from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import pandas as pd
import config

# ---------------------------------------------------------------------------
# Shared state — initialized once at module load, reused across calls
# ---------------------------------------------------------------------------

_encoder = SentenceTransformer(config.ENCODER)

# Cross-encoder re-ranker: a separate model that scores (query, chunk) pairs.
# More accurate than embedding cosine similarity but slower — only run on top-N candidates.
# We retrieve a wider candidate set from Chroma, then re-rank to get the best top_k.
_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

_chroma = chromadb.PersistentClient(path="chroma_db")
_collection = _chroma.get_collection(config.COLLECTION)

CSV_PATH = "data/risk_budget_tracker.csv"

# ---------------------------------------------------------------------------
# Re-ranker
# ---------------------------------------------------------------------------

def rerank(query: str, chunks: list, sources: list, top_n: int = 3):
    """Score each (query, chunk) pair with the cross-encoder, return top_n by score."""
    pairs = [(query, chunk) for chunk in chunks]
    scores = _cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, chunks, sources), key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c, _ in ranked[:top_n]]
    top_sources = [s for _, _, s in ranked[:top_n]]
    return top_chunks, top_sources

# ---------------------------------------------------------------------------
# Tools — the @tool decorator generates the JSON schema from the function
# signature and docstring. The model sees the docstring as the description.
# ---------------------------------------------------------------------------

@tool
def search_knowledge_base(query: str) -> str:
    """Search the local document knowledge base (PDFs, DOCX, HTML) for information
    relevant to the query. Use this for general policy, evaluation, and program questions."""
    q_emb = _encoder.encode(query).tolist()

    # Retrieve more candidates than needed (10), then re-rank down to top_k.
    # Without re-ranking you'd just take the top 3 by cosine similarity, which can miss better matches.
    results = _collection.query(query_embeddings=[q_emb], n_results=10)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    if not chunks:
        return "No relevant documents found."

    top_chunks, top_sources = rerank(query, chunks, sources, top_n=config.TOP_K)

    return "\n---\n".join(f"[{src}]\n{chunk}" for src, chunk in zip(top_sources, top_chunks))


@tool
def query_csv(question: str) -> str:
    """Query the risk and budget tracker spreadsheet for structured data.
    Use this for questions about specific subsystems, risk statuses, budgets,
    schedule status, or quarterly figures — anything that would live in a table."""
    df = pd.read_csv(CSV_PATH)

    # Return column names + full data as a string so the model can reason over it.
    # For small CSVs this is fine. Scaling path: have the model generate pandas code instead.
    summary = (
        f"Columns: {list(df.columns)}\n\n"
        f"Data:\n{df.to_string(index=False)}"
    )
    return summary

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

TOOLS = [search_knowledge_base, query_csv]

SYSTEM_PROMPT = (
    "You are a program analyst assistant with access to two tools:\n"
    "- search_knowledge_base: for general document questions (policy, evaluations, program details)\n"
    "- query_csv: for specific budget, risk status, or quarterly tracker questions\n\n"
    "Always cite source documents in your answer. "
    "If you cannot find the answer in your tools, say so explicitly — "
    "do not answer from memory or outside knowledge."
)

def run_lc_agent(question: str) -> str:
    llm = ChatOllama(model=config.MODEL)

    # The prompt must include {agent_scratchpad} — LangChain injects tool call
    # history there so the model can see what it already tried.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # create_tool_calling_agent binds tools to the model and wires up the prompt.
    # AgentExecutor wraps it in a loop: call tool → get result → call again if needed → stop.
    # verbose=True prints each step — useful for understanding the loop while learning.
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

    result = executor.invoke({"input": question})
    return result["output"]


if __name__ == "__main__":
    question = input("Question: ")
    answer = run_lc_agent(question)
    print("\n" + answer)
