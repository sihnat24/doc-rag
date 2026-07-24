import chromadb
import ollama

from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "program_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.1"
TOP_K = 5

SYSTEM_PROMPT = """You are a program analyst assistant supporting a government equipment modernization program office.
Answer questions using only the provided document excerpts below.
Always cite which document your answer comes from.
If the excerpts do not contain sufficient information to answer the question, respond with:
"I don't have information on that in the available documents."
Do not use outside knowledge."""


def retrieve(question, collection, model, k=TOP_K):
    question_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources


def build_context(chunks, sources):
    sections = []
    for chunk, source in zip(chunks, sources):
        sections.append(f"[Source: {source}]\n{chunk}")
    return "\n\n---\n\n".join(sections)


def answer(question, chunks, sources):
    context = build_context(chunks, sources)
    prompt = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to Chroma...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    print("\nReady. Type your question or 'quit' to exit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        chunks, sources = retrieve(question, collection, model)
        response = answer(question, chunks, sources)

        print(f"\nAnswer:\n{response}\n")
        print("Sources retrieved:", list(dict.fromkeys(sources)))
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
