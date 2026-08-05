# Project Context: Doc-RAG Pipeline

## Purpose
RAG system for DoD-style program office documentation. Answers analyst questions with grounded, cited responses. Explicitly declines when the answer is not in the corpus.

## Stack
- Language: Python (local, no training)
- Embeddings: sentence-transformers `all-MiniLM-L6-v2` (local, free, 384-dim)
- Vector store: Chroma (persistent SQLite at `chroma_db/`)
- Generation LLM: Ollama `llama3.1` (local inference)
- PDF extraction: pdfplumber
- DOCX extraction: python-docx
- HTML extraction: BeautifulSoup
- CSV handling: pandas

## Key Files
- `ingest.py` — document ingestion pipeline
- `agent.py` — agentic query loop (two-call pattern)
- `tools.py` — tool definitions and retrieval logic
- `config.py` — constants (chunk size, top-k, model name, tool schema)
- `main.py` — entry point
- `eval.py` — evaluation runner and scoring
- `data/eval_questions.json` — 20 hand-written eval questions
- `data/eval_results.json` — eval output (auto-scored + manual faithfulness flags)

## Ingestion Pipeline

### Supported File Types and Extraction
| Format | Library | Preprocessing |
|--------|---------|---------------|
| `.pdf` | pdfplumber | Page-by-page extraction, joined with `\n\n` |
| `.docx` | python-docx | Paragraph extraction |
| `.html` | BeautifulSoup | nav/header/footer/script/style tags removed |
| `.csv` | pandas | Rows textified to prose before embedding |

### CSV Textification
Each row converted to: `"Column1: Value1 Column2: Value2 ..."` joined with `\n\n`.
Rationale: raw CSV rows have no grammar; semantic search is unreliable on them. Prose enables embedding similarity.

### Chunking
- Strategy: sliding window on raw extracted text
- Chunk size: 500 characters
- Overlap: 50 characters
- Config key: `CHUNK_SIZE`, `CHUNK_OVERLAP` in `config.py`

### Embedding and Storage
- Embed all chunks offline with `all-MiniLM-L6-v2`
- Store in Chroma collection `"curated_files"` with metadata: `source` (filename)
- Chunk IDs: `{filename}_{chunk_index}`

## Retrieval
- Top-k: 3 (`TOP_K` in `config.py`)
- Similarity: cosine (Chroma default)
- No re-ranking, no hybrid retrieval
- Query embedded at runtime with same model

## Agentic Layer

### Tool
- Name: `search_knowledge_base`
- Input: `query` (string)
- Defined in `config.py` as an Ollama-compatible tool schema

### Two-Call Pattern
1. **Call 1 (decision):** Model receives question + system prompt instructing it to always use `search_knowledge_base`. Model decides whether to invoke tool.
2. **If no tool call:** Return hard decline string — `"I don't have information on that in the available documents."`
3. **If tool called:** Retrieve top-k chunks, format context as `[source]\nchunk` blocks.
4. **Call 2 (generation):** Model receives original messages + tool response + answer system prompt. Generates cited answer.

### System Prompts
- **TOOL_PROMPT:** Instructs model to always use tool before answering; do not answer from memory.
- **ANSWER_PROMPT:** Instructs model to answer only from retrieved excerpts, always cite source, decline explicitly if excerpts don't contain the answer.

## Evaluation

### Question Buckets (20 total)
| Bucket | Count | IDs | Correct behavior |
|--------|-------|-----|-----------------|
| Single-doc lookup | 10 | Q01–Q10 | Answer found in one document |
| Cross-doc synthesis | 6 | Q11–Q16 | Answer requires combining multiple docs |
| Trap questions | 4 | Q17–Q20 | Answer not in corpus → must explicitly decline |

### Metrics
| Metric | Type | How |
|--------|------|-----|
| Retrieval accuracy | Automated | Recall@k: fraction of expected sources retrieved |
| Trap handling | Automated | Binary: does answer contain decline string? |
| Generation faithfulness | Manual | Flagged in results as `answer_faithful: null` for hand review |

### Eval Output Schema (`eval_results.json`)
```json
{
  "question_id": "Q01",
  "question": "...",
  "expected_sources": ["filename.pdf"],
  "retrieved_sources": ["filename.pdf"],
  "answer": "...",
  "recall": 1.0,
  "trap_handled": null,
  "answer_faithful": null
}
```

## Dataset

### Authored Corpus (ground truth — 6 docs)
| File | Type | Content |
|------|------|---------|
| `TEMP_doc.pdf` | PDF | Dense technical evaluation report (test/schedule/requirements) |
| `eval_report.pdf` | PDF | Vendor evaluation report (scores, past performance, supply chain risks) |
| `risk_budget_tracker.csv` | CSV | Risk status + budget tracker by quarter and subsystem |
| `status_brief.docx` | DOCX | Program status slides (low-grammar bullet content) |
| `risk_management_policy.html` | HTML | Risk management policy (noisy nav/footer removed) |
| `contractor_performance_policy.html` | HTML | CPAR rating policy |

### Noise Layer (~3 unrelated research PDFs)
Dropped into same index unmodified. Stress-tests retrieval precision against distractors.
- `attention.pdf` — NLP research paper
- `image generators are generalist vision leaners.pdf` — computer vision paper
- `koala.pdf` — additional research paper

## Design Decisions (for context)

### Why RAG over fine-tuning
Knowledge changes frequently; answers must be traceable to source documents. Fine-tuning suits stable domains where style/behavior consistency matters more than fresh, citable facts.

### Why textify CSV instead of structured query
Simple v1 approach — keeps all formats in one pipeline. Noted limitation: unreliable for frequently-updated datasets. v2 suggestion: give agent a separate pandas/text-to-SQL tool for spreadsheets.

### Why decline explicitly
High-stakes use case (DoD-style). Hallucinations are unacceptable. Trap question bucket is the most important eval bucket.

## What Is NOT Built (explicitly out of scope)
- Hybrid retrieval (BM25 + embeddings)
- Re-ranking after initial retrieval
- Structured-query tool for spreadsheets
- Formal retrieval eval metrics (precision@k)
- Fine-tuning or pretraining
