import time
import logging
from pathlib import Path
from typing import Dict, Optional
from src.llm import OllamaClient
from src.retrieval import retrieve
from src.agents.planner import plan_queries
from src.agents.multivers_verifier import MultiVerSVerifier
from src.config import TOP_K_RERANK, ROOT
from src.trace import emit_if_active

log = logging.getLogger(__name__)

DEFAULT_CKPT = ROOT / "checkpoints" / "scifact.ckpt"

def run_agentic_multivers(claim: str, llm: OllamaClient, config: Optional[Dict] = None) -> Dict:
    cfg = config or {}
    ckpt = Path(cfg.get("multivers_ckpt", DEFAULT_CKPT))
    t0 = time.time()
    emit_if_active("pipeline_start", input={"claim": claim, "config": cfg}, meta={"pipeline": "agentic_v8", "ckpt": ckpt.name})
    tp = time.time()
    queries = [claim] if cfg.get("no_planner") else plan_queries(claim, llm)
    emit_if_active("planner", input={"claim": claim}, output={"queries": queries}, latency_ms=round((time.time()-tp)*1000,1))
    tr = time.time()
    docs, rmeta = retrieve(
        claim=claim, queries=queries, top_k_final=TOP_K_RERANK,
        use_rerank=not cfg.get("no_rerank", False),
        use_live_fallback=False,
    )
    emit_if_active("retrieve", input={"queries": queries},
                   output={"top_k_docs": [{"doc_id": d["doc_id"], "source": d.get("source"),
                                           "score": d.get("score"), "rerank_score": d.get("rerank_score"),
                                           "title": d.get("title", "")[:200]} for d in docs]},
                   latency_ms=round((time.time()-tr)*1000,1), meta=rmeta)
    tv = time.time()
    verifier = MultiVerSVerifier.get(str(ckpt))
    res = verifier.predict_aggregate(claim, docs)
    emit_if_active("verifier", input={"claim": claim, "n_docs": len(docs)},
                   output={"verdict": res["verdict"], "best_supp_prob": res.get("best_supp_prob"),
                           "best_ref_prob": res.get("best_ref_prob"), "cited_ids": res.get("cited_ids", []),
                           "per_doc": [{"doc_id": p["doc_id"], "pred": p["pred"], "probs": p["probs"]} for p in res["per_doc"]]},
                   latency_ms=round((time.time()-tv)*1000,1), meta={"ckpt": ckpt.name})
    return {
        "claim": claim, "verdict": res["verdict"],
        "rationale": "MultiVerS label-head prediction (SciFact-FT Longformer-large-4096)",
        "cited_ids": res.get("cited_ids", []),
        "best_supp_prob": res.get("best_supp_prob", 0.0),
        "best_ref_prob": res.get("best_ref_prob", 0.0),
        "per_doc": [{"doc_id": p["doc_id"], "pred": p["pred"], "probs": p["probs"]}
                    for p in res["per_doc"]],
        "docs": [{"doc_id": d["doc_id"], "title": d.get("title", ""),
                  "abstract": (d.get("abstract") or "")[:500]} for d in docs],
        "queries": queries, "retrieval_meta": rmeta,
        "parse_error": False,
        "time_s": time.time() - t0, "pipeline": "agentic_multivers",
        "verifier_ckpt": ckpt.name,
    }

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) > 1:
        for line in open(sys.argv[1]):
            r = json.loads(line)
            print(f"[{r['ts_rel_s']:>7.3f}s] {r['node']:18s} ({r.get('latency_ms','-')}ms)")
            for k in ("input", "output", "meta"):
                if r.get(k):
                    print(f"    {k}: {json.dumps(r[k])[:300]}")
