import json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm import OllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("pillar2")

JUDGE_MODEL = "qwen2.5:7b-instruct-q4_K_M"

def deepeval_faithfulness(preds, limit=180):
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    from deepeval.models import DeepEvalBaseLLM

    class OllamaJudge(DeepEvalBaseLLM):
        def __init__(self, model): self.model = model
        def load_model(self): return self
        def generate(self, prompt, schema=None):
            cli = OllamaClient(model=self.model, timeout=120)
            if schema is not None:
                out = cli.json_call(prompt, max_tokens=512)
                if out is None: return schema()
                try: return schema.model_validate(out)
                except: return schema()
            return cli.generate(prompt, temperature=0.0, max_tokens=512)
        async def a_generate(self, prompt, schema=None): return self.generate(prompt, schema)
        def get_model_name(self): return self.model

    judge = OllamaJudge(JUDGE_MODEL)
    results = []
    for i, p in enumerate(preds[:limit]):
        if not p.get("docs"):
            results.append({"claim_id": p["claim_id"], "faithfulness": None, "answer_relevancy": None})
            continue
        ctx = [f"{d.get('title','')}. {d.get('abstract','')}" for d in p["docs"][:5]]
        tc = LLMTestCase(
            input=p["claim"],
            actual_output=p.get("rationale") or p.get("verdict", ""),
            retrieval_context=ctx,
        )
        fm = FaithfulnessMetric(model=judge, threshold=0.5, async_mode=False)
        fm.measure(tc)
        fs = float(fm.score) if fm.score is not None else None

        am = AnswerRelevancyMetric(model=judge, threshold=0.5, async_mode=False)
        am.measure(tc)
        as_ = float(am.score) if am.score is not None else None

        results.append({"claim_id": p["claim_id"], "faithfulness": fs, "answer_relevancy": as_})
        if (i+1) % 20 == 0:
            log.info("  %s %d/%d", preds[0].get("claim_id","")[0:10], i+1, min(limit, len(preds)))
    return results

def main():
    pipelines = ["agentic_v7", "standard_rag", "llm_only"]
    summary = {}

    for pipe in pipelines:
        pred_path = Path(f"results/predictions/{pipe}/sm_synth/sm.jsonl")
        out_rows = Path(f"results/pillar2/{pipe}_rows.jsonl")
        out_json = Path(f"results/pillar2/{pipe}_summary.json")

        preds = [json.loads(l) for l in open(pred_path) if l.strip()]
        log.info("=== %s: %d preds ===", pipe, len(preds))

        done = {}
        if out_rows.exists():
            for l in open(out_rows):
                try:
                    r = json.loads(l)
                    done[r["claim_id"]] = r
                except: pass
            log.info("  resuming: %d already done", len(done))

        rows = []
        for p in preds:
            cid = p["claim_id"]
            if cid in done:
                rows.append(done[cid])
            else:
                ctx = [f"{d.get('title','')}. {d.get('abstract','')}" for d in (p.get("docs") or [])[:5]]
                tc_input = p["claim"]
                tc_output = p.get("rationale") or p.get("verdict", "")

                from deepeval.test_case import LLMTestCase
                from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
                from deepeval.models import DeepEvalBaseLLM

                class OllamaJudge(DeepEvalBaseLLM):
                    def __init__(self, model): self.model = model
                    def load_model(self): return self
                    def generate(self, prompt, schema=None):
                        cli = OllamaClient(model=self.model, timeout=120)
                        if schema is not None:
                            out = cli.json_call(prompt, max_tokens=512)
                            if out is None: return schema()
                            try: return schema.model_validate(out)
                            except: return schema()
                        return cli.generate(prompt, temperature=0.0, max_tokens=512)
                    async def a_generate(self, prompt, schema=None): return self.generate(prompt, schema)
                    def get_model_name(self): return self.model

                judge = OllamaJudge(JUDGE_MODEL)
                tc = LLMTestCase(input=tc_input, actual_output=tc_output, retrieval_context=ctx)

                fm = FaithfulnessMetric(model=judge, threshold=0.5, async_mode=False)
                fm.measure(tc)
                fs = float(fm.score) if fm.score is not None else None

                am = AnswerRelevancyMetric(model=judge, threshold=0.5, async_mode=False)
                am.measure(tc)
                as_ = float(am.score) if am.score is not None else None

                row = {"claim_id": cid, "faithfulness": fs, "answer_relevancy": as_}
                rows.append(row)
                done[cid] = row
                with open(out_rows, "a") as f:
                    f.write(json.dumps(row) + "\n")

                if len(rows) % 20 == 0:
                    log.info("  %s %d/%d", pipe, len(rows), len(preds))

        valid_f = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
        valid_a = [r["answer_relevancy"] for r in rows if r["answer_relevancy"] is not None]
        result = {
            "n": len(rows),
            "faithfulness_mean": sum(valid_f)/len(valid_f) if valid_f else None,
            "faithfulness_n": len(valid_f),
            "answer_relevancy_mean": sum(valid_a)/len(valid_a) if valid_a else None,
            "answer_relevancy_n": len(valid_a),
        }
        out_json.write_text(json.dumps(result, indent=2))
        summary[pipe] = result
        log.info("[%s] faithfulness=%.3f answer_relevancy=%.3f",
                 pipe, result["faithfulness_mean"], result["answer_relevancy_mean"])

    Path("results/pillar2/summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
