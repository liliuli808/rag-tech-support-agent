import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = "/home/jichi/rag-tech-support-agent/artifacts/vector_store"
docs = json.load(open(os.path.join(BASE, "documents.json"), encoding="utf-8"))
index = faiss.read_index(os.path.join(BASE, "faiss.index"))
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("chunks in index :", index.ntotal)
print("dimensions      :", index.d)
print()

for q in [
    "What does error E-2011 mean and how do I fix it?",
    "The office router is not responding",
]:
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    sims, ids = index.search(qv, 4)
    print("=" * 72)
    print("Q:", q)
    for r, (s, i) in enumerate(zip(sims[0], ids[0]), 1):
        sec = docs[i]["metadata"].get("section", "?")
        txt = " ".join(docs[i]["page_content"].split())
        print("  #%d  sim=%.4f  [%s]" % (r, s, sec))
        print("      " + txt[:160])
    print()
