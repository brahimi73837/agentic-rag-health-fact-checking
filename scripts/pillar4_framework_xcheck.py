import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import OllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("pillar4")

JUDGE_MODEL = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_BASE = "http://127.0.0.1:11434"

def run_deepeval(preds, limit):
    try:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (FaithfulnessMetric, ContextualRelevancyMetric,
                                       AnswerRelevancyMetric)
        from deepeval.models import DeepEvalBaseLLM
    except Exception as e:
        log.error("deepeval import failed: %s", e); return []

    class OllamaJudge(DeepEvalBaseLLM):
        def __init__(self, model): self.model = model
        def load_model(self): return self
        def generate(self, prompt, schema=None):
            cli = OllamaClient(model=self.model, timeout=120)
            if schema is not None:
                out = cli.json_call(prompt, max_tokens=512)
                if out is None: return schema()
                try: return schema.model_validate(out)
                except Exception: return schema()
            return cli.generate(prompt, temperature=0.0, max_tokens=512)
        async def a_generate(self, prompt, schema=None): return self.generate(prompt, schema)
        def get_model_name(self): return self.model

    judge = OllamaJudge(JUDGE_MODEL)
    rows = []
    for p in preds[:limit]:
        if not p.get("docs"): continue
        ctx = [f"{d.get('title','')}. {d.get('abstract','')}" for d in p["docs"][:5]]
        tc = LLMTestCase(
            input=p["claim"],
            actual_output=p.get("rationale") or p.get("verdict", ""),
            retrieval_context=ctx,
        )
        scores = {"claim_id": p["claim_id"]}
        for cls, key in [(FaithfulnessMetric, "faithfulness"),
                         (ContextualRelevancyMetric, "contextual_relevancy"),
                         (AnswerRelevancyMetric, "answer_relevancy")]:
            try:
                m = cls(model=judge, threshold=0.5, async_mode=False)
                m.measure(tc); scores[key] = float(m.score) if m.score is not None else None
            except Exception as e:
                log.warning("deepeval %s failed for %s: %s", key, p["claim_id"], e)
                scores[key] = None
        rows.append(scores)
        log.info("deepeval %s: %s", p["claim_id"], {k: v for k, v in scores.items() if k != "claim_id"})
    return rows

def run_openevals(preds, limit):
    try:
        from openevals.llm import create_llm_as_judge
        from openevals.prompts import (RAG_GROUNDEDNESS_PROMPT,
                                        RAG_RETRIEVAL_RELEVANCE_PROMPT)
    except Exception as e:
        log.error("openevals import failed: %s", e); return []

    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    os.environ.setdefault("OPENAI_BASE_URL", f"{OLLAMA_BASE}/v1")

    grounded = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT, feedback_key="groundedness",
        model=f"openai:{JUDGE_MODEL}",
    )
    relevance = create_llm_as_judge(
        prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT, feedback_key="retrieval_relevance",
        model=f"openai:{JUDGE_MODEL}",
    )

    rows = []
    for p in preds[:limit]:
        if not p.get("docs"): continue
        ctx = "\n\n".join(f"{d.get('title','')}. {d.get('abstract','')}" for d in p["docs"][:5])
        out = {"claim_id": p["claim_id"]}
        try:
            r = grounded(inputs=p["claim"],
                         outputs=p.get("rationale") or p.get("verdict", ""),
                         context=ctx)
            out["groundedness"] = float(r.get("score", 0)) if r else None
        except Exception as e:
            log.warning("openevals grounded fail %s: %s", p["claim_id"], e); out["groundedness"] = None
        try:
            r = relevance(inputs=p["claim"], context=ctx)
            out["retrieval_relevance"] = float(r.get("score", 0)) if r else None
        except Exception as e:
            log.warning("openevals relevance fail %s: %s", p["claim_id"], e); out["retrieval_relevance"] = None
        rows.append(out)
        log.info("openevals %s: %s", p["claim_id"], {k: v for k, v in out.items() if k != "claim_id"})
    return rows

def run_ragas_pillar4(preds, limit, out_csv):
    from src.eval.ragas_runner import run_ragas
    return run_ragas(preds, JUDGE_MODEL, out_csv, n_subset=limit)

def pearson(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3: return None
    n = len(pairs); xs, ys = zip(*pairs)
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in pairs)
    dx = sum((x-mx)**2 for x in xs)**0.5; dy = sum((y-my)**2 for y in ys)**0.5
    return num/(dx*dy) if dx and dy else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    preds = [json.loads(l) for l in open(args.preds) if l.strip()]
    log.info("loaded %d preds", len(preds))

    log.info("=== DeepEval ===")
    de_rows = run_deepeval(preds, args.limit)
    log.info("=== openevals ===")
    oe_rows = run_openevals(preds, args.limit)
    log.info("=== RAGAS subset ===")
    ragas_csv = out_path.parent / f"ragas_{out_path.stem}.csv"
    ragas_summary = run_ragas_pillar4(preds, args.limit, ragas_csv)

    de_by = {r["claim_id"]: r for r in de_rows}
    oe_by = {r["claim_id"]: r for r in oe_rows}
    ids = [r["claim_id"] for r in de_rows if r["claim_id"] in oe_by]
    de_faith = [de_by[i].get("faithfulness") for i in ids]
    oe_grnd = [oe_by[i].get("groundedness") for i in ids]
    de_ctx = [de_by[i].get("contextual_relevancy") for i in ids]
    oe_rel = [oe_by[i].get("retrieval_relevance") for i in ids]

    correlations = {
        "faithfulness_deepeval__vs__openevals_groundedness": pearson(de_faith, oe_grnd),
        "ctx_relevancy_deepeval__vs__openevals_retrieval_relevance": pearson(de_ctx, oe_rel),
    }

    summary = {
        "n_claims": len(ids),
        "ragas_summary": ragas_summary,
        "deepeval_means": {
            k: (sum(v for v in [r.get(k) for r in de_rows] if v is not None) /
                max(1, sum(1 for v in [r.get(k) for r in de_rows] if v is not None)))
            for k in ("faithfulness", "contextual_relevancy", "answer_relevancy")
        },
        "openevals_means": {
            k: (sum(v for v in [r.get(k) for r in oe_rows] if v is not None) /
                max(1, sum(1 for v in [r.get(k) for r in oe_rows] if v is not None)))
            for k in ("groundedness", "retrieval_relevance")
        },
        "pearson_correlations": correlations,
    }

    out_path.write_text(json.dumps(summary, indent=2, default=str))
    (out_path.parent / f"deepeval_rows_{out_path.stem}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in de_rows))
    (out_path.parent / f"openevals_rows_{out_path.stem}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in oe_rows))

    print(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", out_path)

if __name__ == "__main__":
    main()
