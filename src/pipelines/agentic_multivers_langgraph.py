from langgraph.graph import StateGraph, END

from src.llm import OllamaClient
from src.retrieval import retrieve
from src.agents.planner import plan_queries
from src.agents.multivers_verifier import MultiVerSVerifier
from src.config import TOP_K_RERANK, ROOT
from pathlib import Path

class AgenticMultiversState(dict):
    claim: str
    config: dict
    llm: OllamaClient = None
    queries: list = []
    docs: list = []
    retrieval_meta: dict = {}
    verifier_output: dict = {}
    final_verdict: str = "NEI"
    cited_ids: list = []
    best_supp_prob: float = 0.0
    best_ref_prob: float = 0.0
    per_doc: list = []
    parse_error: bool = False
    time_s: float = 0.0

def node_plan(state: AgenticMultiversState) -> AgenticMultiversState:
    import time
    t0 = time.time()
    cfg = state.get("config", {})
    if cfg.get("no_planner"):
        queries = [state["claim"]]
    else:
        queries = plan_queries(state["claim"], state["llm"])
    state["queries"] = queries
    state["_timing"] = state.get("_timing", {})
    state["_timing"]["plan_s"] = round(time.time() - t0, 3)
    return state

def node_retrieve(state: AgenticMultiversState) -> AgenticMultiversState:
    import time
    t0 = time.time()
    cfg = state.get("config", {})
    docs, rmeta = retrieve(
        claim=state["claim"],
        queries=state["queries"],
        top_k_final=TOP_K_RERANK,
        use_rerank=not cfg.get("no_rerank", False),
        use_live_fallback=False,
    )
    state["docs"] = docs
    state["retrieval_meta"] = rmeta
    state["_timing"]["retrieve_s"] = round(time.time() - t0, 3)
    return state

def node_verify(state: AgenticMultiversState) -> AgenticMultiversState:
    import time
    t0 = time.time()
    cfg = state.get("config", {})
    ckpt = Path(cfg.get("multivers_ckpt", ROOT / "checkpoints" / "scifact.ckpt"))
    verifier = MultiVerSVerifier.get(str(ckpt))
    res = verifier.predict_aggregate(state["claim"], state["docs"])

    state["verifier_output"] = res
    state["final_verdict"] = res.get("verdict", "NEI")
    state["cited_ids"] = res.get("cited_ids", [])
    state["best_supp_prob"] = res.get("best_supp_prob", 0.0)
    state["best_ref_prob"] = res.get("best_ref_prob", 0.0)
    state["per_doc"] = [{"doc_id": p["doc_id"], "pred": p["pred"], "probs": p["probs"]}
                        for p in res.get("per_doc", [])]
    state["parse_error"] = False
    state["_timing"]["verify_s"] = round(time.time() - t0, 3)
    return state

def _build_graph() -> StateGraph:
    g = StateGraph(AgenticMultiversState)

    g.add_node("plan", node_plan)
    g.add_node("retrieve", node_retrieve)
    g.add_node("verify", node_verify)

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "verify")
    g.add_edge("verify", END)

    return g.compile()

_multivers_graph = None

def get_multivers_graph():
    global _multivers_graph
    if _multivers_graph is None:
        _multivers_graph = _build_graph()
    return _multivers_graph

def run_agentic_multivers_langgraph(claim: str, llm: OllamaClient, config: dict = None) -> dict:
    import time
    t0 = time.time()
    cfg = config or {}

    initial = AgenticMultiversState(
        claim=claim,
        config=cfg,
        llm=llm,
    )

    graph = get_multivers_graph()
    result = graph.invoke(initial)

    total_s = time.time() - t0

    return {
        "claim": claim,
        "verdict": result.get("final_verdict", "NEI"),
        "rationale": "MultiVerS label-head prediction (SciFact-FT Longformer-large-4096)",
        "cited_ids": result.get("cited_ids", []),
        "best_supp_prob": result.get("best_supp_prob", 0.0),
        "best_ref_prob": result.get("best_ref_prob", 0.0),
        "per_doc": result.get("per_doc", []),
        "docs": [{"doc_id": d["doc_id"], "title": d.get("title", ""),
                  "abstract": (d.get("abstract") or "")[:500]} for d in result.get("docs", [])],
        "queries": result.get("queries", []),
        "retrieval_meta": result.get("retrieval_meta", {}),
        "parse_error": result.get("parse_error", False),
        "time_s": total_s,
        "pipeline": "agentic_multivers_langgraph",
        "verifier_ckpt": Path(cfg.get("multivers_ckpt", ROOT / "checkpoints" / "scifact.ckpt")).name,
        "node_timing": result.get("_timing", {}),
    }
