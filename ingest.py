import os, re, uuid
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
import tiktoken
from openai import OpenAI
import chromadb

load_dotenv()
client = OpenAI()
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# --- chunking helpers ---
enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def chunk_text(text: str, max_tokens=500, overlap=100):
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        window = tokens[i:i+max_tokens]
        chunks.append(enc.decode(window))
        i += max_tokens - overlap
    return chunks

def clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# --- PDF -> pages -> text ---
def pdf_to_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        yield i, clean(raw)

def embed_batches(texts, batch_size=64):
    """
    Embed a list of strings in batches.
    Prints progress like: 'embedded 064/512 chunks'.
    Returns a list of embedding vectors aligned with 'texts'.
    """
    all_vecs = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_vecs.extend([d.embedding for d in resp.data])
        print(f"  embedded {i + len(batch):>5}/{n} chunks", end="\r")
    print()  # newline after the carriage-return prints
    return all_vecs

def ingest_single_pdf(pdf_path: Path, vectordir: str = "chroma", collection_name: str = "edu-tutor"):
    db = chromadb.PersistentClient(path=vectordir)
    col = db.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    print(f"Indexing: {pdf_path.name}")
    ids, docs, metas = [], [], []

    page_count = 0
    for page_no, page_text in pdf_to_pages(pdf_path):
        page_count += 1
        if not page_text:
            continue
        if page_no % 10 == 0:
            print(f"  parsed up to page {page_no}")
        for idx, chunk in enumerate(chunk_text(page_text, max_tokens=500, overlap=80)):
            ids.append(f"{pdf_path.name}-p{page_no}-{idx}-{uuid.uuid4().hex[:8]}")
            docs.append(chunk)
            metas.append({"source": pdf_path.name, "page": page_no})

    print(f"  Total chunks to embed: {len(docs)}")
    if not docs:
        return {"pages": page_count, "chunks": 0}

    vectors = embed_batches(docs, batch_size=64)

    add_batch = 256
    total = len(docs)
    for i in range(0, total, add_batch):
        col.add(
            ids=ids[i:i+add_batch],
            documents=docs[i:i+add_batch],
            metadatas=metas[i:i+add_batch],
            embeddings=vectors[i:i+add_batch],
        )
        print(f"  added {min(i+add_batch, total):>5}/{total} to Chroma", end="\r")
    print()

    return {"pages": page_count, "chunks": total}

def main():
    data_dir = Path("data")
    for pdf in data_dir.glob("*.pdf"):
        ingest_single_pdf(pdf)

if __name__ == "__main__":
    try:
        _ = client.embeddings.create(model=EMBED_MODEL, input=["hello"])
        print("OpenAI embeddings OK ✅")
    except Exception as e:
        raise SystemExit(f"Embedding preflight failed ❌: {e}")
    main()

