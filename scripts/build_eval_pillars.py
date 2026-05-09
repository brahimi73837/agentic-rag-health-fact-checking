from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm import OllamaClient

EXP_DIR = ROOT / "results" / "experiments"
COMBINED = ROOT / "results" / "eval_pillars"
COMBINED.mkdir(parents=True, exist_ok=True)

LABELS = ["SUPPORTED", "REFUTED", "NEI"]

EXPERIMENTS = {
    "exp1_scifact":      ["agentic", "agentic_multivers", "standard_rag", "llm_only"],
    "exp2_social_media": ["agentic", "standard_rag", "llm_only"],
    "exp3_synthetic":    ["agentic", "agentic_nopost", "standard_rag", "llm_only"],
}

RETRIEVAL_SYSTEMS = {"agentic", "agentic_multivers", "agentic_nopost", "standard_rag"}

def load_preds(exp: str, system: str) -> List[Dict[str, Any]]:
    path = EXP_DIR / exp / "predictions" / f"{system}.jsonl"
    if not path.exists():
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out

def context_text(p: Dict[str, Any], k: int = 5) -> str:
    docs = p.get("docs") or []
    blocks = []
    for i, d in enumerate(docs[:k]):
        title = (d.get("title") or "").strip()
        abstract = str(d.get("abstract") or "").strip()[:600]
        pmid = d.get("doc_id") or d.get("pmid") or ""
        blocks.append(f"[{i + 1}] (PMID {pmid}) {title}\n{abstract}")
    return "\n\n".join(blocks) if blocks else "(no evidence retrieved)"

def cohen_kappa(a: List[str], b: List[str]) -> Optional[float]:
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe >= 0.999:
        return None
    return round((po - pe) / (1 - pe), 4)

def observed_agreement(a: List[str], b: List[str]) -> Optional[float]:
    n = len(a)
    return round(sum(1 for x, y in zip(a, b) if x == y) / n, 4) if n else None

def pearson(xs: List[Optional[float]], ys: List[Optional[float]]) -> Optional[float]:
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pts)
    if n < 3:
        return None
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    syy = sum(y * y for _, y in pts)
    sxy = sum(x * y for x, y in pts)
    den = math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    return round((n * sxy - sx * sy) / den, 4) if den else None

def mean(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None

def pillar1_system(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs = [(p.get("gold_label"), p.get("verdict")) for p in preds
             if p.get("gold_label") and p.get("verdict")]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    idx = {l: i for i, l in enumerate(LABELS)}
    cm = [[0, 0, 0] for _ in LABELS]
    for g, v in pairs:
        if g in idx and v in idx:
            cm[idx[g]][idx[v]] += 1
    per_class, f1s = {}, []
    for i, lab in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(3)) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lab] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
        f1s.append(f1)
    golds = [g for g, _ in pairs]
    preds_l = [v for _, v in pairs]
    return {
        "accuracy": round(sum(1 for g, v in pairs if g == v) / n, 4),
        "macro_f1": round(sum(f1s) / 3, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": LABELS,
        "cohens_kappa": cohen_kappa(golds, preds_l),
        "n": n,
    }

P2_SYS = ("You are a careful biomedical evaluator scoring a fact-checking "
          "rationale against the evidence it was given. Return JSON only.")

P2_PROMPT = """CLAIM: {claim}

RETRIEVED EVIDENCE:
{context}

SYSTEM RATIONALE: {rationale}

Score two properties on a 0.0-1.0 scale:
- faithfulness: every statement in the rationale is supported by the evidence above (1.0 = fully supported, 0.0 = unsupported/contradicted).
- answer_relevancy: the rationale actually addresses the claim rather than a neighbouring topic.

Return JSON: {{"faithfulness": <float>, "answer_relevancy": <float>}}"""

def _clip01(x: Any) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return None

def pillar2_row(llm: OllamaClient, p: Dict[str, Any]) -> Dict[str, Any]:
    if not p.get("docs"):
        return {"claim_id": p.get("claim_id"), "faithfulness": None, "answer_relevancy": None}
    out = llm.json_call(
        P2_PROMPT.format(claim=p.get("claim", ""), context=context_text(p),
                         rationale=p.get("rationale") or p.get("verdict", "")),
        system=P2_SYS, temperature=0.0, max_tokens=200)
    out = out or {}
    return {"claim_id": p.get("claim_id"),
            "faithfulness": _clip01(out.get("faithfulness")),
            "answer_relevancy": _clip01(out.get("answer_relevancy"))}

P3_SYS = "You are one of three independent evidence-groundedness judges. Return JSON only."

P3_RUBRICS = {
    "coverage": "How much of the rationale's wording is actually found in the retrieved abstracts? "
                "Score 1.0 if the rationale is fully covered by the evidence text, 0.0 if none of it is.",
    "citation": "Are the PMIDs the system cited actually present in the retrieved evidence, and do they "
                "back the verdict? Score 1.0 if every cited PMID is present and supportive, 0.0 if cited "
                "PMIDs are absent or irrelevant.",
    "strength": "Does the verdict match how strongly the best-matching abstract supports or refutes the "
                "claim? Score 1.0 if the verdict strength matches the evidence strength, 0.0 if it does not.",
}

P3_PROMPT = """CLAIM: {claim}
VERDICT: {verdict}
CITED PMIDS: {cited}

RETRIEVED EVIDENCE:
{context}

SYSTEM RATIONALE: {rationale}

Judging rubric: {rubric}

Return JSON: {{"score": <float 0.0-1.0>, "vote": "yes|no|abstain"}}
where vote = yes if grounded, no if not grounded, abstain if undecidable."""

def pillar3_judge(llm: OllamaClient, p: Dict[str, Any], rubric_name: str) -> Tuple[Optional[float], str]:
    out = llm.json_call(
        P3_PROMPT.format(
            claim=p.get("claim", ""), verdict=p.get("verdict", ""),
            cited=", ".join(str(c) for c in (p.get("cited_ids") or [])) or "(none)",
            context=context_text(p), rationale=p.get("rationale") or p.get("verdict", ""),
            rubric=P3_RUBRICS[rubric_name]),
        system=P3_SYS, temperature=0.0, max_tokens=160) or {}
    vote = str(out.get("vote", "")).lower().strip()
    if vote not in ("yes", "no", "abstain"):
        vote = "abstain"
    return _clip01(out.get("score")), vote

def pillar3_system(llm: OllamaClient, preds: List[Dict[str, Any]]) -> Tuple[Dict, List[Dict]]:
    judges = ["coverage", "citation", "strength"]
    scores = {j: [] for j in judges}
    votes = {j: [] for j in judges}
    rows = []
    for p in preds:
        s_row, v_row = {}, {}
        for j in judges:
            s, v = pillar3_judge(llm, p, j)
            scores[j].append(s)
            votes[j].append(v)
            s_row[j], v_row[j] = s, v
        cnt = Counter(v_row.values())
        top, tn = cnt.most_common(1)[0]
        rows.append({"claim_id": p.get("claim_id"), "verdict": p.get("verdict"),
                     "gold": p.get("gold_label"), "judge_scores": s_row,
                     "judge_votes": v_row, "majority": top if tn >= 2 else "abstain"})
    n = len(rows)
    maj = [r["majority"] for r in rows]
    pr = {f"{a}__{b}": pearson(scores[a], scores[b]) for a, b in combinations(judges, 2)}
    pa = {f"{a}__{b}": observed_agreement(votes[a], votes[b]) for a, b in combinations(judges, 2)}
    pk = {f"{a}__{b}": cohen_kappa(votes[a], votes[b]) for a, b in combinations(judges, 2)}
    prs = [v for v in pr.values() if v is not None]
    ags = [v for v in pa.values() if v is not None]
    ks = [v for v in pk.values() if v is not None]
    summary = {
        "n": n,
        "majority_yes_rate": round(maj.count("yes") / n, 4) if n else 0.0,
        "majority_no_rate": round(maj.count("no") / n, 4) if n else 0.0,
        "majority_abstain_rate": round(maj.count("abstain") / n, 4) if n else 0.0,
        "per_judge_yes_rate": {j: round(votes[j].count("yes") / n, 4) if n else 0.0 for j in judges},
        "pairwise_pearson_r": pr,
        "pairwise_observed_agreement": pa,
        "pairwise_kappa": pk,
        "mean_pairwise_pearson_r": round(sum(prs) / len(prs), 4) if prs else None,
        "mean_pairwise_observed_agreement": round(sum(ags) / len(ags), 4) if ags else None,
        "mean_pairwise_kappa": round(sum(ks) / len(ks), 4) if ks else None,
    }
    return summary, rows

def pillar4_system(preds: List[Dict[str, Any]], judge_model: str, limit: int) -> Tuple[Dict, Dict]:
    from scripts.pillar4_framework_xcheck import run_deepeval, run_openevals
    from src.eval.ragas_runner import run_ragas

    subset = [p for p in preds if p.get("docs")][:limit] if limit else [p for p in preds if p.get("docs")]
    de_rows = run_deepeval(subset, len(subset))
    oe_rows = run_openevals(subset, len(subset))
    ragas_summary = run_ragas(subset, judge_model, COMBINED / "_ragas_tmp.csv", n_subset=len(subset))

    de_by = {r["claim_id"]: r for r in de_rows}
    oe_by = {r["claim_id"]: r for r in oe_rows}
    ids = [r["claim_id"] for r in de_rows if r["claim_id"] in oe_by]
    de_faith = [de_by[i].get("faithfulness") for i in ids]
    oe_grnd = [oe_by[i].get("groundedness") for i in ids]
    de_ctx = [de_by[i].get("contextual_relevancy") for i in ids]
    oe_rel = [oe_by[i].get("retrieval_relevance") for i in ids]
    corr = {
        "faithfulness__deepeval_vs_openevals": pearson(de_faith, oe_grnd),
        "context__deepeval_vs_openevals": pearson(de_ctx, oe_rel),
    }
    rs = [v for v in corr.values() if v is not None]
    summary = {
        "n_claims": len(ids),
        "ragas_summary": ragas_summary,
        "deepeval_means": {k: mean([r.get(k) for r in de_rows])
                           for k in ("faithfulness", "contextual_relevancy", "answer_relevancy")},
        "openevals_means": {k: mean([r.get(k) for r in oe_rows])
                            for k in ("groundedness", "retrieval_relevance")},
        "pearson_correlations": corr,
        "mean_pearson_r": round(sum(rs) / len(rs), 4) if rs else None,
    }
    return summary, {"deepeval": de_rows, "openevals": oe_rows}

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

def run(judge_model: str, limit: int, do_pillar4: bool):
    llm = OllamaClient(model=judge_model)
    reliability = {"per_experiment": {}}
    all_p3r, all_agree, all_r = [], [], []

    for exp, systems in EXPERIMENTS.items():
        print(f"=== {exp} ===")
        out_dir = EXP_DIR / exp / "pillars"
        out_dir.mkdir(parents=True, exist_ok=True)
        preds_by_sys = {s: load_preds(exp, s) for s in systems}
        systems = [s for s in systems if preds_by_sys[s]]

        p1 = {"per_system": {s: pillar1_system(preds_by_sys[s]) for s in systems}}
        (out_dir / "pillar1.json").write_text(json.dumps(p1, indent=2))

        p2: Dict[str, Any] = {}
        p3: Dict[str, Any] = {"systems": {}, "judges": ["coverage", "citation", "strength"]}
        p4: Dict[str, Any] = {"systems": {}}
        exp_block = {"pillar3_judge_pearson": {}, "pillar3_observed_agreement": {},
                     "pillar4_framework_pearson": {}}

        for s in systems:
            if s not in RETRIEVAL_SYSTEMS:
                continue
            preds = preds_by_sys[s]

            rows2 = [pillar2_row(llm, p) for p in preds]
            write_jsonl(out_dir / f"pillar2_rows_{s}.jsonl", rows2)
            p2[s] = {"n": len(rows2),
                     "faithfulness_mean": mean([r["faithfulness"] for r in rows2]),
                     "answer_relevancy_mean": mean([r["answer_relevancy"] for r in rows2])}

            s3, rows3 = pillar3_system(llm, preds)
            write_jsonl(out_dir / f"pillar3_rows_{s}.jsonl", rows3)
            p3["systems"][s] = s3

            if do_pillar4:
                s4, rows4 = pillar4_system(preds, judge_model, limit)
                for fw, rws in rows4.items():
                    write_jsonl(out_dir / f"pillar4_{fw}_rows_{s}.jsonl", rws)
                p4["systems"][s] = s4
                r4 = s4["mean_pearson_r"]
            else:
                r4 = None

            pr3 = s3["mean_pairwise_pearson_r"]
            ag3 = s3["mean_pairwise_observed_agreement"]
            exp_block["pillar3_judge_pearson"][s] = pr3
            exp_block["pillar3_observed_agreement"][s] = ag3
            exp_block["pillar4_framework_pearson"][s] = r4
            if pr3 is not None:
                all_p3r.append(pr3)
            if ag3 is not None:
                all_agree.append(ag3)
            if r4 is not None:
                all_r.append(r4)
            print(f"  {s:18s} acc={p1['per_system'][s].get('accuracy')} "
                  f"p2_faith={p2[s]['faithfulness_mean']} p3_judgeR={pr3} p4_r={r4}")

        (out_dir / "pillar2.json").write_text(json.dumps(p2, indent=2))
        (out_dir / "pillar3.json").write_text(json.dumps(p3, indent=2))
        (out_dir / "pillar4.json").write_text(json.dumps(p4, indent=2))
        reliability["per_experiment"][exp] = exp_block

    reliability["overall_retrieval_systems_only"] = {
        "pillar3_mean_judge_pearson_r": round(sum(all_p3r) / len(all_p3r), 4) if all_p3r else None,
        "pillar3_mean_observed_agreement": round(sum(all_agree) / len(all_agree), 4) if all_agree else None,
        "pillar4_mean_framework_pearson_r": round(sum(all_r) / len(all_r), 4) if all_r else None,
    }
    (COMBINED / "reliability_summary.json").write_text(json.dumps(reliability, indent=2))
    print(json.dumps(reliability["overall_retrieval_systems_only"], indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen2.5:7b-instruct-q4_K_M")
    ap.add_argument("--limit", type=int, default=0, help="cap claims for pillar 4 (0 = all)")
    ap.add_argument("--no-pillar4", action="store_true", help="skip the framework cross-check")
    args = ap.parse_args()
    run(args.judge, args.limit, not args.no_pillar4)

if __name__ == "__main__":
    main()
