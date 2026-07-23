# Doc-RAG + Agent Project

## Scenario
Program analyst supporting a DoD-style program office modernization effort. Years of heterogeneous docs (PDFs, decks, spreadsheets, policy pages) — none searchable together. Build a system that answers leadership questions with grounded, cited responses and explicitly declines when docs don't cover the question.

## Architecture Decisions

**Foundation model + RAG** (not fine-tuning, not pretraining)
- Claude API for generation (only spend tokens on final answer call)
- sentence-transformers for local embeddings (free)
- Chroma for persisted vector store

**Why RAG over fine-tuning:** Knowledge changes often, answers need to be traceable/citable. Fine-tuning suits stable domains where style/behavior consistency matters more than fresh, citable facts.

**Spreadsheet handling:** Raw CSV rows have no grammar — semantic search is unreliable on them. Strategy: textify rows into prose before embedding (e.g. "In Q3, VendorA's program had a risk status of Yellow with $500,000 allocated.") so they flow through the same pipeline as prose docs.
- **v2 improvement noted:** Give agent a separate structured-query tool (pandas/text-to-SQL) for spreadsheets; let agent choose retrieval path. This is the scaling answer for frequently-updated datasets.

**Agentic layer:** Model has a `search_knowledge_base` tool and decides whether to retrieve or explicitly decline — not a fixed retrieve-then-generate script.

## Dataset Plan

**Primary corpus (self-authored, 4–6 docs) — ground truth is known:**
- Dense-prose PDF-style technical evaluation report
- Bullet-fragment "slide deck" style doc (tests chunking on low-grammar content)
- Spreadsheet/CSV risk-and-budget tracker (textified before embedding)
- Webpage-style reference doc (with nav/footer noise to handle)

**Noise layer:** ~12 unrelated research papers from Kaggle "RAG-Practice" set (medical imaging, land-cover classification, NLP, etc.). Dropped into same index unmodified — stress-test retrieval precision against distractors.

## Eval Plan (~15–20 hand-written questions, 3 buckets)

1. **Single-doc lookup** — answer in one doc. Tests basic retrieval + generation.
2. **Cross-doc synthesis** — answer requires combining two docs. Tests multi-hit retrieval.
3. **Trap questions** — answer doesn't exist in corpus. Correct behavior = explicit "I don't have information on that." Most important bucket given stakes.

**Two metrics scored separately:**
- Retrieval accuracy: did pipeline pull the right chunks?
- Generation faithfulness: given right chunks, did answer stick to them?

## Build Plan

1. Draft 4–6 authored docs + 15–20 eval questions (questions first, docs built to answer them)
2. Pull noise-layer PDFs from Kaggle set
3. Ingestion pipeline: extract text per format → chunk → local embeddings → Chroma
4. Plain RAG query script: embed question → retrieve top-k → Claude API generates cited answer
5. Agentic layer: `search_knowledge_base` tool, model decides retrieve vs. decline
6. Run eval set against both versions, score by hand
7. Write up findings

## Stack
- Python (local, no training)
- sentence-transformers (embeddings)
- Chroma (vector store)
- Claude API (generation)
- PyMuPDF or pdfplumber (PDF extraction)
- pandas (CSV/spreadsheet handling)

## "What I'd Explore Next" (explicitly NOT building, for interview context)
- Hybrid retrieval (BM25 + embeddings) for keyword-heavy queries
- Re-ranking step after initial retrieval
- Structured-query tool for spreadsheets instead of textify-and-embed
- Formal retrieval eval metrics (precision@k) instead of manual scoring

## MITRE Posting Mapping
| Posting language | What this demonstrates |
|---|---|
| "LLM technology applications" | Claude API for grounded generation |
| "prototype... generative AI... agentic AI" | Tool-calling layer, retrieve-vs-decline decision |
| "MLOps/LLMOps" | Local embedding pipeline, persisted vector store, ingest/query separation |
| "knowledge representation and extraction" | Chunking design, format-specific preprocessing |
| "evaluate technology solutions" | Eval question buckets + retrieval vs. faithfulness scoring |
| "convey complex ideas to non-technical audiences" | Plain-language explanations of RAG, textify, agent declination |
