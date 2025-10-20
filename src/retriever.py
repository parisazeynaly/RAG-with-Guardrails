import os
import json
import glob
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "./.rag_index")
MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def load_embedder():
    return SentenceTransformer(MODEL_NAME)

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start <= 0: break
    return chunks

def build_index(docs_path: str) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    files = glob.glob(os.path.join(docs_path, "**", "*.txt"), recursive=True)
    embedder = load_embedder()
    texts, metadatas = [], []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            txt = fh.read()
        for ch in chunk_text(txt):
            texts.append(ch)
            metadatas.append({"source": f})
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "texts.json"), "w", encoding="utf-8") as ftxt:
        json.dump(texts, ftxt, ensure_ascii=False)
    with open(os.path.join(INDEX_DIR, "metadatas.json"), "w", encoding="utf-8") as fmeta:
        json.dump(metadatas, fmeta, ensure_ascii=False)
    print(f"Indexed {len(texts)} chunks from {len(files)} files -> {INDEX_DIR}")

def search(query: str, k: int = 5) -> List[Tuple[str, dict, float]]:
    embedder = load_embedder()
    qemb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qemb)
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "texts.json"), "r", encoding="utf-8") as ftxt:
        texts = json.load(ftxt)
    with open(os.path.join(INDEX_DIR, "metadatas.json"), "r", encoding="utf-8") as fmeta:
        metas = json.load(fmeta)
    D, I = index.search(qemb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append((texts[idx], metas[idx], float(score)))
    return results
