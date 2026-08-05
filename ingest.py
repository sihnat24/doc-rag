#helper functions and main funciton to ingest data 

#given a root dataset, recurse into it and process all possible

#datasets of accepted type

#all helpers just return a block of text



import pandas as pd
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import config

from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def main(data_dir: str, collection_name: str):

    encoder = SentenceTransformer(config.ENCODER)

    paths = get_files(data_dir, config.DATA_TYPES)

    #setup chromadb
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(collection_name)

    chunks = []
    sources = []
    source_ids = []


    print(len(paths))
    for p in paths:
        text = extract_text(p)
        file_chunks = extract_chunks(text)
        for idx, chunk in enumerate(file_chunks):
            source_ids.append(f"{p.name}_{idx}")
            chunks.append(chunk)
            sources.append(p.name)

    embeddings = encoder.encode(chunks).tolist()
    collection.add(ids=source_ids, documents=chunks, embeddings=embeddings, metadatas=[{"source": s} for s in sources])


if __name__ == "__main__":
    main(config.DATA_DIR, config.COLLECTION)
    





