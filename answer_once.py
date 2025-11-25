import os, chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

CHAT_MODEL  = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

db = chromadb.PersistentClient(path="chroma")
col = db.get_or_create_collection("edu-tutor")

def embed(q: str):
    return client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding

def retrieve(q: str, k: int = 6):
    qvec = embed(q)
    # ⬇️ remove "ids" from include
    res = col.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    # ⬇️ ids are returned by default even if not in include
    ids   = res.get("ids", [[]])[0]

    # tiny de-dup by (source,page) so overlap twins don’t spam context
    uniq, out_docs, out_metas, out_ids = set(), [], [], []
    for d, m, _id in zip(docs, metas, ids):
        if not d or not m:  # guard against any None
            continue
        key = (m.get("source"), m.get("page"))
        if key in uniq:
            continue
        uniq.add(key)
        out_docs.append(d); out_metas.append(m); out_ids.append(_id)
        if len(out_docs) == k:
            break
    return out_docs, out_metas, out_ids