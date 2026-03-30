import time
from typing import Dict
from src.llm import OllamaClient
from src.retrieval import retrieve
from src.agents.verifier import verify
from src.config import TOP_K_RERANK

def run_standard_rag(claim: str, llm: OllamaClient) -> Dict:
    t0 = time.time()
    docs, meta = retrieve(
        claim=claim, queries=[claim], top_k_final=TOP_K_RERANK,
        use_rerank=False, use_live_fallback=False,
    )
    v = verify(claim, docs, llm)
    return {
        "claim": claim, "verdict": v["verdict"], "rationale": v.get("rationale", ""),
        "cited_ids": v.get("cited_ids", []),
        "docs": [{"doc_id": d["doc_id"], "title": d.get("title", ""),
                  "abstract": d.get("abstract", "")[:500]} for d in docs],
        "queries": [claim], "retrieval_meta": meta,
        "parse_error": v.get("parse_error", False),
        "time_s": time.time() - t0, "pipeline": "standard_rag",
    }
