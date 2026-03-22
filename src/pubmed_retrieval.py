import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sqlite3
import logging
from typing import List, Dict
import numpy as np
import faiss
import zstandard as zstd

from src.config import PUBMED_FAISS, PUBMED_PMIDS, PUBMED_DB, PUBMED_NPROBE
from src.embeddings import encode_queries

log = logging.getLogger(__name__)

_P = {"idx": None, "pmids": None, "con": None, "dctx": None}

def available() -> bool:
    return PUBMED_FAISS.exists() and PUBMED_PMIDS.exists() and PUBMED_DB.exists()

def _load():
    if _P["idx"] is None:
        _P["idx"] = faiss.read_index(str(PUBMED_FAISS))
        _P["idx"].nprobe = PUBMED_NPROBE
        _P["pmids"] = np.load(PUBMED_PMIDS, mmap_mode="r")
        _P["con"] = sqlite3.connect(f"file:{PUBMED_DB}?mode=ro", uri=True, check_same_thread=False)
        _P["dctx"] = zstd.ZstdDecompressor()
    return _P["idx"], _P["pmids"], _P["con"], _P["dctx"]

def _fetch_text(con, dctx, pmid: int):
    row = con.execute("SELECT blob FROM docs WHERE pmid=?", (pmid,)).fetchone()
    if not row:
        return None
    return json.loads(dctx.decompress(row[0]))

def pubmed_search(queries: List[str], k: int = 30) -> List[List[Dict]]:
    if not available():
        return [[] for _ in queries]
    idx, pmids, con, dctx = _load()
    q = encode_queries(queries)
    faiss.normalize_L2(q)
    D, I = idx.search(q, k)
    out = []
    for qi in range(len(queries)):
        row = []
        for rank, (pos, sc) in enumerate(zip(I[qi], D[qi])):
            if pos < 0 or pos >= len(pmids):
                continue
            pmid = int(pmids[pos])
            doc = _fetch_text(con, dctx, pmid)
            if not doc:
                continue
            row.append({
                "doc_id": f"pubmed:{pmid}",
                "title": doc.get("t", "") or "",
                "abstract": doc.get("a", "") or "",
                "score": float(sc),
                "source": "pubmed_full",
            })
        out.append(row)
    return out
