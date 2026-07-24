#helper functions and main funciton to ingest data 

#given a root dataset, recurse into it and process all possible

#datasets of accepted type

#all helpers just return a block of text


import os 

import pandas as pd
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = 'data'
DATA_TYPES = ['.docx','.html','.pdf','.csv']
ENCODER = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


COLLECTION = "curated_files"


def parse_pdf(path: str) -> str:
    full_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text= page.extract_text()
            if text:
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


def extract_chunks(text: str, size: int, overlap: int) -> list[str]:
    l = len(text)
    p = 0
    chunks = []
    while p + size - overlap < l:
        chunks.append(text[p: p + size])
        p += size - overlap

    if p < l:
        chunks.append(text[p:])
    return chunks


def main():

    encoder = SentenceTransformer(ENCODER)
    
    paths = get_files(DATA_DIR, DATA_TYPES)

    #setup chromadb
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(COLLECTION)

    chunks = []
    sources = []
    source_ids = []


    print(len(paths))
    for p in paths:
        text = extract_text(p)
        file_chunks = extract_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk in enumerate(file_chunks):
            source_ids.append(f"{p.name}_{idx}")
            chunks.append(chunk)
            sources.append(p.name)

    embeddings = encoder.encode(chunks).tolist()
    collection.add(ids=source_ids, documents=chunks, embeddings=embeddings, metadatas=[{"source": s} for s in sources])


if __name__ == "__main__":
    main()
    





