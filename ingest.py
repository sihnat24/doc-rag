#helper functions and main funciton to ingest data 

#given a root dataset, recurse into it and process all possible

#datasets of accepted type

#all helpers just return a block of text



import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import chromadb
import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import config


# --- Ingestion log helpers ---

def load_log() -> dict:
    log_path = Path(config.INGEST_LOG)
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return {}

def save_log(log: dict):
    with open(config.INGEST_LOG, "w") as f:
        json.dump(log, f, indent=2)

def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def already_ingested(path: Path, log: dict) -> bool:
    entry = log.get(path.name)
    if not entry:
        return False
    return entry["hash"] == file_hash(path)

def log_file(path: Path, log: dict, file_type: str):
    log[path.name] = {
        "hash": file_hash(path),
        "ingested_at": datetime.now().isoformat(),
        "type": file_type,
    }


# --- Parsers ---

def detect_columns(page, pixel_thresh: int) -> int:
    words = page.extract_words()
    #build a set of x1 values, look for a gap 
    
    #each word spans x0 to x1, we create a rang of its from x0 to x1 for all words. then look if any xs are missing
    #words is a list of dicts
    #[ {'x0': 72, 'x1': 134, 'top': 100, 'bottom': 112, 'text': 'peripheral'}, ] 

    word_ranges = set()
    for word in words:
        for x in range(int(word['x0']), int(word['x1']) + 1):
            word_ranges.add(x)

    width = page.width

    mid_col = set()
    for i in range(int(0.3 * width), int(0.7 * width)):
        if i not in word_ranges:
            mid_col.add(i)

    if len(mid_col) > pixel_thresh:
        return 2
    return 1


def parse_pdf(path: str) -> str:
    full_text = []
    with pdfplumber.open(path) as pdf:
            for page in pdf.pages:

                #first, extract any tables in the page
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        row_text = " | ".join(cell or "" for cell in row)
                        full_text.append(row_text)

                #for text, first determine column format (not all papers are one nice block of text that can be read left to right row by row)
                cols = detect_columns(page, config.PIXEL_THRESH)
                if cols == 1:
                    text= page.extract_text()
                    if text:
                        full_text.append(text)
                else: #2 col academic paper
                    w, h = page.width, page.height
                    left = page.crop((0, 0, w / 2, h)).extract_text() or ""
                    right = page.crop((w / 2, 0, w, h)).extract_text() or ""
                    text = left + "\n" + right
                    if text.strip():
                        full_text.append(text)

    return "\n\n".join(full_text) #very clear a new page starts
 
def parse_docx(path: str) -> str:
    full_text = []
    doc = Document(path)
    for d in doc.paragraphs:
        text= d.text
        if text:
            full_text.append(text)

    return "\n".join(full_text)


def parse_html(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        for tag in soup.find_all(["nav", "header", "footer", "script",    "style"]): 
            tag.decompose()

        full_text = soup.get_text("\n\n", True)

    return full_text

def parse_csv(path) -> str:
    full_text = []
    df = pd.read_csv(path)

    for _ , row in df.iterrows(): #go row by row
        curr_row = [f"{col}: {val}" for col, val in row.items()] #this parses the row, creates col: data
        full_text.append(" ".join(curr_row))

    return "\n\n".join(full_text)


def parse_xlsx(path) -> pd.DataFrame:
    return pd.read_excel(path)


def extract_text(file: Path) -> str:

    if file.suffix == ".csv":
        return parse_csv(file)

    elif file.suffix == ".pdf":
        return parse_pdf(file)

    elif file.suffix == ".html":
        return parse_html(file)

    elif file.suffix == ".docx":
        return parse_docx(file)

    raise TypeError("File type is not supported")



def get_files(data_dir: str, file_types: list) -> list[Path]:

    file = Path(data_dir).rglob("*")

    paths = []

    for f in file:
        if f.suffix in file_types:
            paths.append(f)

    return paths


def extract_chunks(text: str) -> list[str]:

    splitter = RecursiveCharacterTextSplitter(
      chunk_size=config.CHUNK_SIZE,
      chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    
    return chunks


def ingest_tables(paths: list[Path], log: dict, force: bool):
    """Load CSV/XLSX files into SQLite. Each file becomes its own table."""
    con = sqlite3.connect(config.DB_PATH)

    for p in paths:
        if not force and already_ingested(p, log):
            print(f"  [skip] {p.name} (unchanged)")
            continue

        if p.suffix == ".csv":
            df = pd.read_csv(p)
        else:
            df = parse_xlsx(p)

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        table_name = p.stem.lower().replace(" ", "_").replace("-", "_")
        df.to_sql(table_name, con, if_exists="replace", index=False)

        log_file(p, log, "table")
        print(f"  [table] {p.name} → SQLite table '{table_name}' ({len(df)} rows)")

    con.close()


def ingest_prose(paths: list[Path], collection, encoder, log: dict, force: bool):
    """Chunk and embed prose files into Chroma."""
    chunks, sources, source_ids = [], [], []

    for p in paths:
        if not force and already_ingested(p, log):
            print(f"  [skip] {p.name} (unchanged)")
            continue

        text = extract_text(p)
        file_chunks = extract_chunks(text)
        for idx, chunk in enumerate(file_chunks):
            source_ids.append(f"{p.name}_{idx}")
            chunks.append(chunk)
            sources.append(p.name)

        log_file(p, log, "prose")
        print(f"  [prose] {p.name} → {len(file_chunks)} chunks")

    if chunks:
        embeddings = encoder.encode(chunks).tolist()
        collection.add(
            ids=source_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": s} for s in sources],
        )


def main(data_dir: str, collection_name: str, force: bool = False):
    log = load_log()

    prose_paths = get_files(data_dir, config.DATA_TYPES)
    table_paths = get_files(data_dir, config.TABLE_TYPES)

    print(f"\nFound {len(prose_paths)} prose file(s), {len(table_paths)} table file(s)")

    print("\n-- Tables (SQLite) --")
    ingest_tables(table_paths, log, force)

    print("\n-- Prose (Chroma) --")
    encoder = SentenceTransformer(config.ENCODER)
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(collection_name)
    ingest_prose(prose_paths, collection, encoder, log, force)

    save_log(log)
    print("\nDone. Log saved.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    if force:
        confirm = input("Re-ingest all files? This will overwrite existing data. [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)
    main(config.DATA_DIR, config.COLLECTION, force=force)
    





