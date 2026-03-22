import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import hashlib
import logging
from typing import List, Dict, Tuple
import numpy as np
import faiss

from src.config import (
    FAISS_INDEX, DOC_IDS_NPY, CORPUS_META,
    TOP_K_RETRIEVE, TOP_K_RERANK, LIVE_FALLBACK_SCORE_THRESHOLD, CACHE_DIR,
    USE_PUBMED_POOL, PUBMED_TOP_K,
)
from src.embeddings import encode_queries, rerank_cross

log = logging.getLogger(__name__)

_INDEX = {"faiss": None, "ids": None, "meta": None}
_RETR_CACHE = CACHE_DIR / "retrieval"
_RETR_CACHE.mkdir(parents=True, exist_ok=True)

def _load():
    if _INDEX["faiss"] is None:
        _INDEX["faiss"] = faiss.read_index(str(FAISS_INDEX))
        _INDEX["ids"] = np.load(DOC_IDS_NPY, allow_pickle=True)
        meta = {}
        with open(CORPUS_META) as f:
            for line in f:
                r = json.loads(line)
                meta[r["doc_id"]] = r
        _INDEX["meta"] = meta
    return _INDEX["faiss"], _INDEX["ids"], _INDEX["meta"]

def _ckey(*parts) -> str:
    return hashlib.sha1("||".join(str(p) for p in parts).encode()).hexdigest()[:20]

def faiss_search(queries: List[str], k: int = TOP_K_RETRIEVE) -> List[List[Dict]]:
    idx, ids, meta = _load()
    q_emb = encode_queries(queries)
    faiss.normalize_L2(q_emb)
    scores, I = idx.search(q_emb, k)
    results = []
    for qi in range(len(queries)):
        row = []
        for rank, (did_pos, sc) in enumerate(zip(I[qi], scores[qi])):
            if did_pos < 0:
                continue
            doc_id = str(ids[did_pos])
            d = meta.get(doc_id)
            if d is None:
                continue
            row.append({
                "doc_id": doc_id, "title": d["title"], "abstract": d["abstract"],
                "score": float(sc), "source": d["source"],
            })
        results.append(row)
    return results

def merge_dedup(results_per_query: List[List[Dict]]) -> List[Dict]:
    best = {}
    for row in results_per_query:
        for d in row:
            prev = best.get(d["doc_id"])
            if prev is None or d["score"] > prev["score"]:
                best[d["doc_id"]] = d
    return sorted(best.values(), key=lambda x: -x["score"])

def rerank(claim: str, docs: List[Dict], top_k: int = TOP_K_RERANK) -> List[Dict]:
    if not docs:
        return []
    texts = [f"{d['title']}. {d['abstract']}" for d in docs]
    scores = rerank_cross(claim, texts)
    for d, s in zip(docs, scores):
        d["rerank_score"] = float(s)
    return sorted(docs, key=lambda x: -x["rerank_score"])[:top_k]

def retrieve(
    claim: str,
    queries: List[str],
    top_k_final: int = TOP_K_RERANK,
    use_rerank: bool = True,
    use_live_fallback: bool = True,
    use_pubmed_pool: bool = USE_PUBMED_POOL,
) -> Tuple[List[Dict], Dict]:
    per_q = faiss_search(queries, k=TOP_K_RETRIEVE)
    pools = [per_q]
    n_pubmed = 0
    if use_pubmed_pool:
        from src.pubmed_retrieval import pubmed_search, available as pm_available
        if pm_available():
            pm = pubmed_search(queries, k=PUBMED_TOP_K)
            pools.append(pm)
            n_pubmed = sum(len(r) for r in pm)
        else:
            log.warning("USE_PUBMED_POOL set but pubmed index files missing")
    pool_tops = []
    for pool in pools:
        d = merge_dedup(pool)
        pool_tops.append(d[:TOP_K_RETRIEVE])
    seen = {}
    for d in [x for sub in pool_tops for x in sub]:
        if d["doc_id"] not in seen:
            seen[d["doc_id"]] = d
    merged = list(seen.values())
    meta = {"n_candidates": len(merged),
            "offline_top_score": pool_tops[0][0]["score"] if pool_tops[0] else 0.0,
            "used_live": False, "n_pubmed": n_pubmed}
    if use_live_fallback and (not merged or merged[0]["score"] < LIVE_FALLBACK_SCORE_THRESHOLD):
        from src.ncbi import search_and_fetch
        live_docs = []
        for q in queries[:2]:
            for d in search_and_fetch(q, retmax=10):
                live_docs.append({
                    "doc_id": f"pubmed_live_{d['pmid']}", "title": d["title"],
                    "abstract": d["abstract"], "score": 0.0, "source": "pubmed_live",
                })
        if live_docs:
            merged = merge_dedup([merged, live_docs])
            meta["used_live"] = True
            meta["n_live_added"] = len(live_docs)
    if use_rerank and merged:
        cap = TOP_K_RETRIEVE * max(1, len(pools))
        top = rerank(claim, merged[:cap], top_k=top_k_final)
    else:
        top = merged[:top_k_final]
    return top, meta
