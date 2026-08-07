from langchain.tools import tool
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





