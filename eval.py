import json
import chromadb
from sentence_transformers import SentenceTransformer
import config
import agent

EVAL_PATH = "data/eval_questions.json"
RESULTS_PATH = "data/eval_results.json"
DECLINE_STRING = "I don't have information on that in the available documents."


def recall_at_k(expected_sources: list, retrieved_sources: list) -> float:
    if not expected_sources:
        return 1.0  # trap questions — no sources expected
    hits = sum(1 for src in expected_sources if src in retrieved_sources)
    return hits / len(expected_sources)


def trap_handled(answer: str) -> bool:
    return DECLINE_STRING.lower() in answer.lower()


def main():
    print("Loading encoder and Chroma...")
    encoder = SentenceTransformer(config.ENCODER)
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(config.COLLECTION)

    with open(EVAL_PATH) as f:
        questions = json.load(f)

    results = []
    total = len(questions)
    recall_scores = []
    trap_total = 0
    trap_correct = 0

    print(f"\nRunning eval on {total} questions...\n")
    print("=" * 60)

    for q in questions:
        qid = q["id"]
        bucket = q["bucket"]
        question = q["question"]
        expected_sources = q.get("source_docs", [])
        ground_truth = q.get("answer")
        expected_behavior = q.get("expected_behavior", "")

        print(f"[{qid}] ({bucket}) {question[:80]}...")

        answer, retrieved_sources = agent.run_agent(question, collection, encoder)

        # auto-score retrieval
        recall = recall_at_k(expected_sources, retrieved_sources)
        recall_scores.append(recall)

        # auto-score trap handling
        is_trap = bucket == "trap"
        declined = trap_handled(answer)
        if is_trap:
            trap_total += 1
            if declined:
                trap_correct += 1

        result = {
            "id": qid,
            "bucket": bucket,
            "question": question,
            "ground_truth": ground_truth,
            "expected_behavior": expected_behavior,
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            "answer": answer,
            "recall_at_k": recall,
            "trap_handled": declined if is_trap else None,
            "answer_faithful": None
        }

        results.append(result)

        print(f"  Retrieved: {retrieved_sources}")
        print(f"  Recall@k:  {recall:.2f}")
        if is_trap:
            print(f"  Trap handled: {declined}")
        print(f"  Answer: {answer[:200]}...")
        print()

    avg_recall = sum(recall_scores) / len(recall_scores)
    print("=" * 60)
    print(f"SUMMARY")
    print(f"  Questions run:     {total}")
    print(f"  Avg Recall@k:      {avg_recall:.2f}")
    print(f"  Trap handling:     {trap_correct}/{trap_total} correct")
    print(f"  Faithfulness:      (manual scoring required — see {RESULTS_PATH})")
    print("=" * 60)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
