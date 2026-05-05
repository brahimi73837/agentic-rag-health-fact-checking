import argparse
import json
import logging
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import OllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("pillar3")

JUDGES = ["qwen2.5:7b-instruct-q4_K_M", "llama3.2:3b", "mistral:7b-instruct-q4_0"]

SYS = ("You are a careful biomedical evaluator. Decide whether a fact-checking system's "
       "verdict on a health claim is correct GIVEN the retrieved evidence shown. "
       "Return JSON only.")

PROMPT = """CLAIM: {claim}

RETRIEVED EVIDENCE (top docs system used):
{evidence}

SYSTEM VERDICT: {verdict}
SYSTEM RATIONALE: {rationale}

Question: Given ONLY the evidence above, is the system's verdict correct?
- "yes" — verdict is supported by the evidence.
- "no"  — evidence contradicts or fails to support the verdict.
- "abstain" — evidence insufficient to judge.

Return JSON: {{"judgment": "yes|no|abstain", "reason": "<brief>"}}"""

def fmt_evidence(docs, k=3):
    out = []
    for i, d in enumerate(docs[:k]):
        title = (d.get("title") or "").strip()
        abstract = (d.get("abstract") or "").strip()[:500]
        out.append(f"[{i+1}] {title}\n    {abstract}")
    return "\n".join(out) if out else "(no evidence retrieved)"

def judge_one(model: str, claim: str, docs, verdict: str, rationale: str) -> str:
    llm = OllamaClient(model=model, timeout=120)
    prompt = PROMPT.format(
        claim=claim, evidence=fmt_evidence(docs),
        verdict=verdict, rationale=rationale or "(no rationale)",
    )
    out = llm.json_call(prompt, system=SYS, max_tokens=200)
    if not out: return "abstain"
    j = (out.get("judgment") or "").lower().strip()
    return j if j in ("yes", "no", "abstain") else "abstain"

def cohen_kappa(a, b) -> float:
    assert len(a) == len(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0: return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = Counter(a); pb = Counter(b)
    pe = sum((pa[c] / n) * (pb[c] / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def majority(votes) -> str:
    c = Counter(votes)
    top = c.most_common(1)[0]
    return top[0] if top[1] >= 2 else "abstain"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap claims per system (0=all)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {"systems": {}, "judges": JUDGES}

    for pred_path in args.preds:
        sys_name = Path(pred_path).parents[1].name
        log.info("=== %s (%s) ===", sys_name, pred_path)
        preds = [json.loads(l) for l in open(pred_path)]
        if args.limit: preds = preds[:args.limit]
        per_judge = {j: [] for j in JUDGES}
        majorities, golds, system_verdicts = [], [], []
        rows = []

        for i, p in enumerate(preds):
            claim = p["claim"]; verdict = p.get("verdict", "NEI")
            rationale = p.get("rationale", "")
            docs = p.get("docs", [])
            votes = []
            for jm in JUDGES:
                t0 = time.time()
                v = judge_one(jm, claim, docs, verdict, rationale)
                per_judge[jm].append(v); votes.append(v)
                log.info("[%s][%d/%d %s] %-30s -> %s (%.1fs)",
                         sys_name, i+1, len(preds), p["claim_id"], jm.split(":")[0], v, time.time()-t0)
            mv = majority(votes)
            majorities.append(mv); golds.append(p.get("gold_label"))
            system_verdicts.append(verdict)
            rows.append({"claim_id": p["claim_id"], "verdict": verdict,
                         "gold": p.get("gold_label"), "votes": dict(zip(JUDGES, votes)),
                         "majority": mv})

        kappas = {}
        for a, b in combinations(JUDGES, 2):
            k = cohen_kappa(per_judge[a], per_judge[b])
            kappas[f"{a.split(':')[0]}__{b.split(':')[0]}"] = round(k, 3)

        yes_rate = sum(1 for m in majorities if m == "yes") / max(1, len(majorities))
        no_rate = sum(1 for m in majorities if m == "no") / max(1, len(majorities))
        abstain_rate = sum(1 for m in majorities if m == "abstain") / max(1, len(majorities))

        summary["systems"][sys_name] = {
            "n": len(preds), "majority_yes_rate": round(yes_rate, 3),
            "majority_no_rate": round(no_rate, 3),
            "majority_abstain_rate": round(abstain_rate, 3),
            "pairwise_kappa": kappas,
            "per_judge_yes_rate": {j: round(sum(1 for v in per_judge[j] if v == "yes") / max(1, len(per_judge[j])), 3)
                                   for j in JUDGES},
        }
        rows_path = out_path.parent / f"rows_{sys_name}.jsonl"
        with open(rows_path, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        log.info("[%s] wrote %s", sys_name, rows_path)

    out_path.write_text(json.dumps(summary, indent=2))
    log.info("wrote summary -> %s", out_path)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
