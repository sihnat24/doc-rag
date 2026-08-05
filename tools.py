
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import config


def retrieve(q_emb, k: int, collection: chromadb.Collection) -> tuple:

    results = collection.query(query_embeddings= q_emb, n_results=k)

    chunks = results["documents"][0]
    sources = [m['source'] for m in results["metadatas"][0]]

    return (chunks, sources)


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





