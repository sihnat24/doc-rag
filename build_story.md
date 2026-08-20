# Build Story

A record of what we built, what broke, and why we changed it.
Intended as interview prep — not documentation.

---

## Phase 1: Basic RAG

**What we built:**
- PDF text extraction with pdfplumber → fixed-size chunking → sentence-transformers embeddings → Chroma vector store
- Plain retrieve-then-generate: embed query, pull top-k chunks, pass to LLM

**What we learned:**
- Fixed-size chunking cut sentences and words mid-stream — chunks had no semantic coherence
- All PDFs treated identically regardless of layout — 2-column academic papers read left-to-right across columns, garbling text
- No way to handle structured data (spreadsheets) — semantic search on raw CSV rows is unreliable

---

## Phase 2: Smarter Chunking + Format-Aware Ingestion

**What we changed:**
- Switched to `RecursiveCharacterTextSplitter` — tries paragraph → sentence → word boundaries before falling back to character count
- Added column detection: measure horizontal word coverage, detect gap in middle 40% of page → 2-column layout → crop and extract each column independently
- Added format-specific parsers: PDF, DOCX, HTML (with nav/footer stripping), CSV (textified rows)
- Added hash-based ingest log — skip unchanged files on re-ingest

**What we learned:**
- Recursive chunking helped but didn't eliminate mid-sentence splits — root cause is noisy PDF text, not the splitter
- Column detection worked for body pages but failed on pages with narrow affiliation sidebars (classified as 1-column, sidebar text bled into body sentences)
- Separate table extraction (pdfplumber `extract_tables()`) put all table rows *before* prose on each page, breaking reading order and separating captions from data

---

## Phase 3: Agentic Layer + Structured Query Tool

**What we built:**
- Replaced fixed retrieve-then-generate pipeline with a LangChain agent
- Agent has 4 tools: `search_knowledge_base`, `list_knowledge_base`, `query_spreadsheet`, `web_search`
- Model decides which tool to call — or explicitly declines if the answer isn't in the corpus
- Spreadsheet data ingested into SQLite; agent issues SQL SELECT queries for precise lookups

**Why this matters:**
- Semantic search is unreliable for exact values (Kd = 15 µM, author names, PMIDs)
- SQL gives deterministic answers for structured data; semantic search handles unstructured prose
- Explicit decline behavior is critical for a high-stakes use case — hallucination is worse than "I don't know"

---

## Phase 4: Hybrid Retrieval (BM25 + Dense)

**What we built:**
- Added BM25 index (rank_bm25) built in-memory at startup from all Chroma chunks
- `_hybrid_search()`: runs dense retrieval + BM25 in parallel, merges with Reciprocal Rank Fusion (k=60)
- RRF: each doc scores `1/(k + rank)` from each retriever; scores sum; top-k returned

**Why hybrid:**
- Dense embeddings handle semantic similarity ("what does this paper conclude about EVH1 binding?")
- BM25 handles exact terminology ("PMID 16120600", "Kofler", "NMR", domain acronyms)
- Neither alone covers both failure modes

**Known limitations noted:**
- BM25 uses whitespace tokenization — punctuation attached to words ("Budget." ≠ "Budget") hurts recall
- BM25 index is in-memory, rebuilt at startup — fine at this scale, would need Elasticsearch or SQLite FTS5 at production scale

---

## Phase 5: Corpus + Eval Design

**What we changed:**
- Replaced fabricated DoD program office documents with real scientific literature (GYF/WH1/EVH1 domain papers)
- Domain we actually know → can write precise eval questions, speak to content confidently
- Added Excel metadata tracker (authors, PMIDs, methods, domains) → ingested as SQLite tables
- Multi-sheet Excel handling: each sheet becomes its own SQLite table

**Eval design:**
- 3 buckets: single-doc lookup, cross-doc synthesis, trap questions (answer not in corpus)
- Score retrieval and generation separately — bad answer could be bad retrieval OR bad generation
- Layered test suite: ingest validation → parsing correctness → retrieval accuracy → tool routing → end-to-end

---

## Phase 6: PDF Parsing Overhaul (In Progress)

**What we found:**
- pdfplumber `extract_tables()` and PyMuPDF `find_tables()` both rely on ruled borders — academic paper tables are whitespace-aligned, so detection fails completely
- Removing separate table extraction (letting `extract_text()` handle everything) preserved reading order but tables still chunk mid-row
- 100-char minimum filter removed garbage chunks (`| | |`, page numbers, isolated keywords)
- Sidebar affiliation text (narrow right-column on abstract pages) bleeds into body sentences — root cause: column detection classifies abstract page as 1-column

**Where we landed:**
- Switching to Marker for PDF parsing — layout-aware model trained on scientific papers (PubLayNet)
- Handles 2-column layouts, table structure, figure captions natively
- Outputs structured markdown: cleaner chunking surface, tables preserved as units with captions
- Tradeoff: slower than pdfplumber (runs a layout model per page) → mitigated by ingest-once-persist pattern

---

## 2026-08-19 — Embedding Model Upgrade: MiniLM → MPNet

**Problem:** Even after fixing noise and bumping chunk size to 900, mid-sentence splits persisted. `RecursiveCharacterTextSplitter` tries paragraph → sentence → word boundaries before falling back to character splits, so the splitter itself wasn't the issue — the chunk size ceiling was.

**Root cause:** `all-MiniLM-L6-v2` has a 256-token max sequence length (~1000–1200 chars). Chunks larger than that are silently truncated during embedding — the tail of the chunk is invisible to retrieval entirely. So there was no point pushing chunk size past ~1100.

**Fix:** Switched encoder to `all-mpnet-base-v2` (384-token max, ~1400–1500 chars). Same local/free setup via sentence-transformers, meaningfully better embedding quality, and enough headroom to let the splitter find real sentence boundaries.

- `ENCODER`: `all-MiniLM-L6-v2` → `all-mpnet-base-v2`
- `CHUNK_SIZE`: 900 → 1400
- `CHUNK_OVERLAP`: 100 → 150

**Tradeoff:** MPNet is slower to encode (~3–4x vs MiniLM) and the vectors are 768-dim vs 384-dim, so Chroma uses more memory. Neither matters at this corpus size. Would matter at millions of chunks.

**Unresolved — chunk overlap is still mid-sentence:** Even with larger chunks and clean cut points, the overlap backtrack is dumb — it goes back exactly `CHUNK_OVERLAP` chars regardless of where that lands. So the start of every non-first chunk is likely mid-sentence. The overlap didn't cause the original bad cut, but it reintroduces a mid-sentence fragment at every boundary. The real fix is a sentence-aware splitter (e.g., pack whole sentences up to the size limit, overlap by whole sentences). Not implemented — retrieval still works because the core content is intact. Logged as a known imprecision.

**Table chunks bypass the splitter — embedding truncation risk:** Tables are inserted as pre-formed chunks and never pass through `RecursiveCharacterTextSplitter`. Two tables in Singer 2024 exceeded 2x `CHUNK_SIZE` (~2800 chars), which is well past the encoder's ~1500-char practical limit. Those tables are silently truncated during embedding — rows near the bottom are invisible to retrieval. No fix implemented yet; added a test that flags tables over 2x `CHUNK_SIZE` as a warning.

**Takeaway:** Chunk size and embedding model max length are coupled. Tuning one without knowing the other's ceiling is wasted effort. Always check the model's `max_seq_length` before pushing chunk size.

---

## 2026-08-19 — Marker Chunk Quality: Post-Ingest Debug Session

**Problems found (manual inspection of `tests/singer_chunks.txt`):**

1. **Mid-sentence splits**: `CHUNK_SIZE=500` is too small for academic prose. Sentences commonly run 200–300 chars; with `CHUNK_OVERLAP=50`, splits landed mid-sentence and the overlap caused the partial sentence to repeat at the start of the next chunk.

2. **Front matter noise**: Marker preserves the full paper including author affiliations, correspondence blocks, funding acknowledgments — all of which appear before the Abstract. These became their own chunks with zero retrieval value.

3. **References section noise**: The full bibliography (chunks 186–248 in Singer 2024) was ingested as prose chunks. Citation fragments like "doi.org/10.1093/..." are semantically meaningless and pollute the embedding space.

4. **Residual HTML tags**: `<span id="page-X-0">` tags from Marker's markdown output were only stripped in the TABLE detection path — they bled through into prose lines.

5. **Markdown link syntax**: `[text](url)` rendered as-is in prose chunks, adding URL noise to every citation-heavy sentence.

**How each was handled:**

- **Chunk size**: bumped `CHUNK_SIZE` 500→900, `CHUNK_OVERLAP` 50→100. Fewer mid-sentence breaks; overlap large enough to preserve cross-boundary context.
- **Front matter**: `parse_pdf_marker` now skips all lines until it detects the `## Abstract` heading. Works because Marker consistently outputs an Abstract heading for academic papers.
- **References/Acknowledgments/Author Contributions**: added a `STOP_HEADINGS` regex — once matched, all subsequent lines are discarded. Drops ~30+ junk chunks per paper.
- **HTML tags**: moved the `re.sub(r"<[^>]+>", "", line)` strip to the top of the loop so it applies to all lines, not just TABLE detection.
- **Markdown links**: added `re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean_line)` before appending to prose — keeps anchor text, drops URL.

**Takeaway**: Marker's output quality is high but it preserves the full document structure. Noise filtering has to be explicit — the parser doesn't know what's front matter vs. body vs. bibliography. Section heading detection is the right hook.

---

## What We'd Explore Next (Not Built)

- **Proper BM25 tokenization**: NLTK/spaCy with punctuation stripping instead of whitespace split
- **Persistent BM25**: pickle or SQLite FTS5 instead of in-memory rebuild
- **Re-ranking**: cross-encoder re-rank after initial retrieval (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Retrieval metrics**: precision@k instead of manual scoring
- **Figure content**: Marker outputs figure regions but image content (graphs, gels) requires vision model or OCR to extract data
- **Graph layer**: entity/relationship traversal for "how does X connect to Y" queries — not worth it at this corpus size
