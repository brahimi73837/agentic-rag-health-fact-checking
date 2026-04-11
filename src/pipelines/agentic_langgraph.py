from typing import Literal
from langgraph.graph import StateGraph, END
from collections import Counter

from src.llm import OllamaClient
from src.retrieval import retrieve
from src.agents.planner import plan_queries
from src.agents.verifier import _format_evidence, SYSTEM, PROMPT_TMPL
from src.agents.posthoc_checks import check_direct_support, check_contradiction
from src.config import TOP_K_RERANK, LABELS

class AgenticState(dict):
    claim: str
    config: dict
    llm: OllamaClient = None
    queries: list = []
    docs: list = []
    retrieval_meta: dict = {}
    samples: list = []
    base_verdict: str = "NEI"
    adjustments: list = []
    final_verdict: str = "NEI"
    rationale: str = ""
    cited_ids: list = []
    support_sentence: str = ""
    contra_sentence: str = ""
    parse_error: bool = False
    time_s: float = 0.0

def node_plan(state: AgenticState) -> AgenticState:
    import time
    t0 = time.time()
    cfg = state.get("config", {})
    if cfg.get("no_planner"):
        queries = [state["claim"]]
    else:
        queries = plan_queries(state["claim"], state["llm"])
    state["queries"] = queries
    state["_timing"] = state.get("_timing", {})
    state["_timing"]["plan_s"] = time.time() - t0
    return state

def node_retrieve(state: AgenticState) -> AgenticState:
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
    state["_timing"]["retrieve_s"] = time.time() - t0
    return state

def _verify_at_temp(state: AgenticState, temperature):
    prompt = PROMPT_TMPL.format(claim=state["claim"], evidence_block=_format_evidence(state["docs"]))
    out = state["llm"].json_call(prompt, system=SYSTEM, max_tokens=350, temperature=temperature)
    if not out:
        return {"verdict": "NEI", "rationale": "parse_error", "cited_ids": [], "parse_error": True}
    v = str(out.get("verdict", "")).upper().strip()
    if v not in LABELS:
        v = "NEI"
    return {"verdict": v, "rationale": str(out.get("rationale", ""))[:600],
            "cited_ids": list(out.get("cited_ids") or []), "parse_error": False}

def node_verify(state: AgenticState) -> AgenticState:
    import time
    t0 = time.time()
    samples = [_verify_at_temp(state, t) for t in (0.0, 0.3, 0.6)]
    cnt = Counter(s["verdict"] for s in samples)
    top, n = cnt.most_common(1)[0]
    base_verdict = "NEI" if n == 1 else top
    rep = next((s for s in samples if s["verdict"] == base_verdict and not s.get("parse_error")), samples[0])

    state["samples"] = samples
    state["base_verdict"] = base_verdict
    state["final_verdict"] = base_verdict
    state["rationale"] = rep.get("rationale", "")
    state["cited_ids"] = rep.get("cited_ids", [])
    state["adjustments"] = []
    state["parse_error"] = all(s.get("parse_error") for s in samples)
    state["_timing"]["verify_s"] = time.time() - t0
    return state

def node_posthoc_contradiction(state: AgenticState) -> AgenticState:
    import time
    t0 = time.time()
    if state["base_verdict"] not in ("SUPPORTED", "NEI"):
        state["_timing"]["posthoc_contradiction_s"] = time.time() - t0
        return state

    triggered, sent = check_contradiction(state["claim"], state["docs"], state["llm"])
    if triggered:
        state["final_verdict"] = "REFUTED"
        state["adjustments"].append(f"contra→REFUTED: {sent[:100]}")
        state["contra_sentence"] = sent
    state["_timing"]["posthoc_contradiction_s"] = time.time() - t0
    return state

def node_posthoc_direct_support(state: AgenticState) -> AgenticState:
    import time
    t0 = time.time()
    if state["final_verdict"] != "SUPPORTED":
        state["_timing"]["posthoc_direct_support_s"] = time.time() - t0
        return state

    supported, sent = check_direct_support(state["claim"], state["docs"], state["llm"])
    if not supported:
        state["final_verdict"] = "NEI"
        state["adjustments"].append("no_direct_support→NEI")
        state["support_sentence"] = sent
    state["_timing"]["posthoc_direct_support_s"] = time.time() - t0
    return state

def _build_graph() -> StateGraph:
    g = StateGraph(AgenticState)

    g.add_node("plan", node_plan)
    g.add_node("retrieve", node_retrieve)
    g.add_node("verify", node_verify)
    g.add_node("posthoc_contradiction", node_posthoc_contradiction)
    g.add_node("posthoc_direct_support", node_posthoc_direct_support)

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "verify")

    def posthoc_enabled(state: AgenticState) -> Literal["posthoc_contradiction", "__end__"]:
        cfg = state.get("config", {})
        if cfg.get("no_post_hoc"):
            return "__end__"
        return "posthoc_contradiction"

    g.add_conditional_edges("verify", posthoc_enabled, {
        "posthoc_contradiction": "posthoc_contradiction",
        "__end__": END,
    })

    g.add_edge("posthoc_contradiction", "posthoc_direct_support")
    g.add_edge("posthoc_direct_support", END)

    return g

_agentic_graph = None

def get_agentic_graph():
    global _agentic_graph
    if _agentic_graph is None:
        _agentic_graph = _build_graph().compile()
    return _agentic_graph

def run_agentic_langgraph(claim: str, llm: OllamaClient, config: dict = None) -> dict:
    import time
    t0 = time.time()
    cfg = config or {}

    initial = AgenticState(
        claim=claim,
        config=cfg,
        llm=llm,
    )

    graph = get_agentic_graph()
    result = graph.invoke(initial)

    total_s = time.time() - t0

    return {
        "claim": claim,
        "verdict": result.get("final_verdict", "NEI"),
        "base_verdict": result.get("base_verdict", "NEI"),
        "rationale": result.get("rationale", ""),
        "cited_ids": result.get("cited_ids", []),
        "votes": dict(Counter(s["verdict"] for s in result.get("samples", []))),
        "samples": [{"v": s["verdict"], "err": s.get("parse_error", False)} for s in result.get("samples", [])],
        "adjustments": result.get("adjustments", []),
        "support_sentence": result.get("support_sentence", ""),
        "contra_sentence": result.get("contra_sentence", ""),
        "docs": [{"doc_id": d["doc_id"], "title": d.get("title", ""),
                  "abstract": (d.get("abstract") or "")[:500]} for d in result.get("docs", [])],
        "queries": result.get("queries", []),
        "retrieval_meta": result.get("retrieval_meta", {}),
        "parse_error": result.get("parse_error", False),
        "time_s": total_s,
        "pipeline": "agentic_langgraph",
        "node_timing": {k: round(v, 3) for k, v in result.get("_timing", {}).items()},
    }
