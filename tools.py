
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import config


def retrieve(q_emb: str, k: int, collection: chromadb.Collection) -> tuple:

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


def main():

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(config.COLLECTION)

    encoder = SentenceTransformer()


    print("Input 'quit' to end program")
    
    while True:

        usr_input = input("Question: ")

        if (usr_input.strip().lower()) == "quit":
            break

        usr_emb = encoder.encode(usr_input)

        chunks, sources = retrieve(usr_emb, config.TOP_K, collection)
        model_answer = get_answer(usr_input, chunks, sources)

        unique_sources = list(set(sources))
 
        print(f"Answer: {model_answer}\n\n,sources cited: {(', '.join(unique_sources))}")


if __name__ == "__main__":
    main()





