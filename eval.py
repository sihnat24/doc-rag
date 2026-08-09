import json
import re
from collections import defaultdict

import ollama

import agent
import config

EVAL_PATH = "data/eval_questions.json"
RESULTS_PATH = "data/eval_results.json"
DECLINE_PHRASE = "I don't have information on that in the available documents."

TABLE_EXTENSIONS = {".csv", ".xlsx"}
TABLE_TOOL = "query_spreadsheet"
PROSE_TOOL = "search_knowledge_base"


# --- Scoring helpers ---

def recall_at_k(expected_sources: list, retrieved_sources: list) -> float:
    if not expected_sources:
        return 1.0  # trap questions: no sources expected, full credit
    hits = sum(1 for src in expected_sources if src in retrieved_sources)
    return hits / len(expected_sources)


def trap_handled(answer: str) -> bool:
    return DECLINE_PHRASE.lower() in answer.lower()


def expected_tool(source_docs: list) -> str:
    """Infer which tool should be called based on source file types."""
    for doc in source_docs:
        ext = "." + doc.rsplit(".", 1)[-1].lower() if "." in doc else ""
        if ext in TABLE_EXTENSIONS:
            return TABLE_TOOL
    return PROSE_TOOL


def routing_correct(tools_called: list, source_docs: list, bucket: str) -> bool | None:
    """Did the agent call the right tool for this question type?"""
    if bucket == "trap":
        return None  # routing not scored for trap questions
    target = expected_tool(source_docs)
    return target in tools_called


def parse_sources_from_output(tool_outputs: list) -> list[str]:
    """Extract source filenames from search_knowledge_base tool outputs."""
    sources = []
    for output in tool_outputs:
        # format: "source 1: filename, chunk text."
        matches = re.findall(r"source \d+:\s*([^,]+),", output)
        sources.extend(m.strip() for m in matches)
    return list(set(sources))


def score_faithfulness(question: str, answer: str, tool_outputs: list) -> tuple[bool | None, str]:
    """LLM-as-judge: does the answer stay within the retrieved context?"""
    context = "\n\n".join(tool_outputs).strip()
    if not context:
        return None, "No context retrieved — faithfulness not applicable."

    prompt = (
        f"You are evaluating whether an AI-generated answer is faithful to the retrieved context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Does the answer stay within the information provided in the context, "
        "without adding outside knowledge or hallucinating facts? "
        "Reply with FAITHFUL or UNFAITHFUL, then one sentence explaining why."
    )

    response = ollama.chat(
        model=config.MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response["message"]["content"].strip()
    first_line = text.split("\n")[0].upper()
    faithful = "FAITHFUL" in first_line and "UNFAITHFUL" not in first_line
    return faithful, text


# --- Main ---

def main():
    with open(EVAL_PATH) as f:
        questions = json.load(f)

    results = []
    bucket_stats = defaultdict(lambda: {
        "total": 0, "recall": [], "routing_correct": 0,
        "routing_scored": 0, "faithful": 0, "faithful_scored": 0,
        "trap_total": 0, "trap_correct": 0,
    })

    total = len(questions)
    print(f"\nRunning eval on {total} questions...\n{'=' * 60}")

    for q in questions:
        qid        = q["id"]
        bucket     = q["bucket"]
        question   = q["question"]
        src_docs   = q.get("source_docs", [])
        ground_truth      = q.get("answer")
        expected_behavior = q.get("expected_behavior", "")

        print(f"\n[{qid}] ({bucket}) {question[:80]}...")

        trace  = agent.run_agent_eval(question)
        answer = trace["answer"]
        tools_called  = trace["tools_called"]
        tool_outputs  = trace["tool_outputs"]

        # Retrieval sources (prose tool only)
        retrieved_sources = parse_sources_from_output(tool_outputs)

        # Recall@k
        recall = recall_at_k(src_docs, retrieved_sources)

        # Trap handling
        is_trap = bucket == "trap"
        declined = trap_handled(answer)

        # Tool routing
        routed_ok = routing_correct(tools_called, src_docs, bucket)

        # Faithfulness (LLM-as-judge)
        faithful, faithful_note = score_faithfulness(question, answer, tool_outputs)

        # Accumulate bucket stats
        bs = bucket_stats[bucket]
        bs["total"] += 1
        bs["recall"].append(recall)
        if is_trap:
            bs["trap_total"] += 1
            if declined:
                bs["trap_correct"] += 1
        if routed_ok is not None:
            bs["routing_scored"] += 1
            if routed_ok:
                bs["routing_correct"] += 1
        if faithful is not None:
            bs["faithful_scored"] += 1
            if faithful:
                bs["faithful"] += 1

        result = {
            "id": qid,
            "bucket": bucket,
            "question": question,
            "ground_truth": ground_truth,
            "expected_behavior": expected_behavior,
            "expected_sources": src_docs,
            "retrieved_sources": retrieved_sources,
            "tools_called": tools_called,
            "answer": answer,
            "recall_at_k": recall,
            "routing_correct": routed_ok,
            "trap_handled": declined if is_trap else None,
            "faithful": faithful,
            "faithful_note": faithful_note,
        }
        results.append(result)

        print(f"  Tools called:  {tools_called}")
        print(f"  Retrieved:     {retrieved_sources}")
        print(f"  Recall@k:      {recall:.2f}")
        if routed_ok is not None:
            print(f"  Routing:       {'PASS' if routed_ok else 'FAIL'} (expected {expected_tool(src_docs)})")
        if is_trap:
            print(f"  Trap handled:  {declined}")
        if faithful is not None:
            print(f"  Faithful:      {'YES' if faithful else 'NO'} — {faithful_note[:80]}")
        print(f"  Answer:        {answer[:200]}...")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY BY BUCKET")
    print(f"{'=' * 60}")

    for bucket, bs in bucket_stats.items():
        avg_recall = sum(bs["recall"]) / len(bs["recall"]) if bs["recall"] else 0
        print(f"\n  [{bucket.upper()}]  ({bs['total']} questions)")
        print(f"    Avg Recall@k:   {avg_recall:.2f}")
        if bs["routing_scored"]:
            print(f"    Tool routing:   {bs['routing_correct']}/{bs['routing_scored']} correct")
        if bs["trap_total"]:
            print(f"    Trap handling:  {bs['trap_correct']}/{bs['trap_total']} correct")
        if bs["faithful_scored"]:
            print(f"    Faithfulness:   {bs['faithful']}/{bs['faithful_scored']} faithful")

    print(f"\n{'=' * 60}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Full results → {RESULTS_PATH}\n")


if __name__ == "__main__":
    main()
