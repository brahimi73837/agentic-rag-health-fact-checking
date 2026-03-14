import json
import logging
import numpy as np
import faiss
from tqdm import tqdm

from src.config import (
    CORPUS_JSONL, FAISS_INDEX, DOC_IDS_NPY, CORPUS_META, EMBED_DIM
)
from src.embeddings import encode_articles

log = logging.getLogger(__name__)

def iter_corpus(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

def build_index(batch_size: int = 8, nlist: int = 256):
    recs = list(iter_corpus(CORPUS_JSONL))
    n = len(recs)
    print(f"[index] encoding {n} docs")

    embs = np.zeros((n, EMBED_DIM), dtype="float32")
    for i in tqdm(range(0, n, batch_size)):
        chunk = recs[i:i + batch_size]
        titles = [r["title"] for r in chunk]
        abstracts = [r["abstract"] for r in chunk]
        embs[i:i + len(chunk)] = encode_articles(titles, abstracts, batch_size=batch_size)

    faiss.normalize_L2(embs)

    if n < 2000:
        idx = faiss.IndexFlatIP(EMBED_DIM)
        idx.add(embs)
    else:
        nlist_eff = min(nlist, max(4, int(np.sqrt(n))))
        quant = faiss.IndexFlatIP(EMBED_DIM)
        idx = faiss.IndexIVFFlat(quant, EMBED_DIM, nlist_eff, faiss.METRIC_INNER_PRODUCT)
        idx.train(embs)
        idx.add(embs)
        idx.nprobe = min(16, nlist_eff)

    FAISS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(FAISS_INDEX))
    np.save(DOC_IDS_NPY, np.array([r["doc_id"] for r in recs]))
    with open(CORPUS_META, "w") as w:
        for r in recs:
            w.write(json.dumps(r) + "\n")
    print(f"[index] wrote {FAISS_INDEX} ({n} vectors, dim={EMBED_DIM})")

if __name__ == "__main__":
    build_index()
