import time
import logging
from collections import Counter
from typing import Dict, Optional
from src.llm import OllamaClient
from src.retrieval import retrieve
from src.agents.planner import plan_queries
from src.agents.verifier import verify, _format_evidence, SYSTEM, PROMPT_TMPL
from src.agents.posthoc_checks import check_direct_support, check_contradiction
from src.config import TOP_K_RERANK, LABELS

log = logging.getLogger(__name__)

def _verify_at_temp(claim, docs, llm, temperature):
    prompt = PROMPT_TMPL.format(claim=claim, evidence_block=_format_evidence(docs))
    out = llm.json_call(prompt, system=SYSTEM, max_tokens=800, temperature=temperature)
    if not out:
        return {"verdict": "NEI", "rationale": "parse_error", "cited_ids": [], "parse_error": True}
    v = str(out.get("verdict", "")).upper().strip()
    if v not in LABELS:
        v = "NEI"
    return {"verdict": v, "rationale": str(out.get("rationale", ""))[:600],
            "cited_ids": list(out.get("cited_ids") or []), "parse_error": False}

def run_agentic(claim: str, llm: OllamaClient, config: Optional[Dict] = None) -> Dict:
    cfg = config or {}
    t0 = time.time()
    queries = [claim] if cfg.get("no_planner") else plan_queries(claim, llm)
    docs, rmeta = retrieve(
        claim=claim, queries=queries, top_k_final=TOP_K_RERANK,
        use_rerank=not cfg.get("no_rerank", False),
        use_live_fallback=False,
    )
    samples = [_verify_at_temp(claim, docs, llm, t) for t in (0.0, 0.3, 0.6)]
    cnt = Counter(s["verdict"] for s in samples)
    top, n = cnt.most_common(1)[0]
    base_verdict = "NEI" if n == 1 else top
    rep = next((s for s in samples if s["verdict"] == base_verdict and not s.get("parse_error")),
               samples[0])

    adjustments = []
    final_verdict = base_verdict

    contra_triggered = False
    contra_sent = ""
    if base_verdict in ("SUPPORTED", "NEI"):
        contra_triggered, contra_sent = check_contradiction(claim, docs, llm)
        if contra_triggered:
            final_verdict = "REFUTED"
            adjustments.append(f"contra→REFUTED: {contra_sent[:100]}")

    support_direct = True
    support_sent = ""
    if final_verdict == "SUPPORTED":
        support_direct, support_sent = check_direct_support(claim, docs, llm)
        if not support_direct:
            final_verdict = "NEI"
            adjustments.append("no_direct_support→NEI")

    return {
        "claim": claim, "verdict": final_verdict,
        "base_verdict": base_verdict,
        "rationale": rep.get("rationale", ""),
        "cited_ids": rep.get("cited_ids", []),
        "votes": dict(cnt),
        "samples": [{"v": s["verdict"], "err": s.get("parse_error", False)} for s in samples],
        "adjustments": adjustments,
        "support_sentence": support_sent,
        "contra_sentence": contra_sent,
        "docs": [{"doc_id": d["doc_id"], "title": d.get("title", ""),
                  "abstract": (d.get("abstract") or "")[:500]} for d in docs],
        "queries": queries, "retrieval_meta": rmeta,
        "parse_error": all(s.get("parse_error") for s in samples),
        "time_s": time.time() - t0, "pipeline": "agentic",
    }
