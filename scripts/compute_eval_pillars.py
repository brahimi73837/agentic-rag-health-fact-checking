from __future__ import annotations

import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "results" / "experiments"
COMBINED_DIR = ROOT / "results" / "eval_pillars"
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["SUPPORTED", "REFUTED", "NEI"]

EXPERIMENTS = {
    "exp1_scifact": ["agentic", "agentic_multivers", "standard_rag", "llm_only"],
    "exp2_social_media": ["agentic", "standard_rag", "llm_only"],
    "exp3_synthetic": ["agentic", "agentic_nopost", "standard_rag", "llm_only"],
}

STOPWORDS = set("""
a an the and or but if then else of to in on at by for with without from into over under
is are was were be been being do does did has have had this that these those it its as
than so such not no nor can could should would may might will shall we our us you your
they their them he she his her i me my which who whom whose what when where why how
study studies finding findings result results evidence claim claims show shows showed
suggest suggests suggested associated association risk effect effects patients group
""".split())

WORD_RE = re.compile(r"[a-z0-9]+")

def _norm_abstract(abstract: Any) -> str:
    if isinstance(abstract, list):
        return " ".join(str(x) for x in abstract)
    return str(abstract or "")

def doc_text(d: Dict[str, Any]) -> str:
    return f"{d.get('title','')}. {_norm_abstract(d.get('abstract',''))}".strip()

def tokens(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())

def content_tokens(text: str) -> List[str]:
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 2]

def sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [s.strip() for s in parts if len(s.strip()) > 3]

def jaccard(a: str, b: str) -> float:
    sa, sb = set(content_tokens(a)), set(content_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def token_f1(a: str, b: str) -> float:
    ca, cb = Counter(content_tokens(a)), Counter(content_tokens(b))
    if not ca or not cb:
        return 0.0
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(ca.values())
    rec = overlap / sum(cb.values())
    return 2 * prec * rec / (prec + rec)

class TfidfSpace:

    def __init__(self, corpus: List[str]):
        self.idf: Dict[str, float] = {}
        n = max(1, len(corpus))
        df: Counter = Counter()
        for doc in corpus:
            for t in set(content_tokens(doc)):
                df[t] += 1
        for t, d in df.items():
            self.idf[t] = math.log((n + 1) / (d + 1)) + 1.0

    def vec(self, text: str) -> Dict[str, float]:
        tf = Counter(content_tokens(text))
        if not tf:
            return {}
        return {t: (c / sum(tf.values())) * self.idf.get(t, 1.0) for t, c in tf.items()}

    def cosine(self, a: str, b: str) -> float:
        va, vb = self.vec(a), self.vec(b)
        if not va or not vb:
            return 0.0
        common = set(va) & set(vb)
        num = sum(va[t] * vb[t] for t in common)
        na = math.sqrt(sum(v * v for v in va.values()))
        nb = math.sqrt(sum(v * v for v in vb.values()))
        if na == 0 or nb == 0:
            return 0.0
        return num / (na * nb)

def cohen_kappa(a: List[str], b: List[str]) -> Optional[float]:
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe >= 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 4)

def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 3:
        return None
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    syy = sum(y * y for _, y in pts)
    sxy = sum(x * y for x, y in pts)
    num = n * sxy - sx * sy
    den = math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    if den == 0:
        return None
    return round(num / den, 4)

def mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None

def load_preds(exp: str, system: str) -> List[Dict[str, Any]]:
    path = EXP_DIR / exp / "predictions" / f"{system}.jsonl"
    if not path.exists():
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out

def pillar1_system(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs = [(p.get("gold_label"), p.get("verdict")) for p in preds
             if p.get("gold_label") and p.get("verdict")]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    acc = sum(1 for g, v in pairs if g == v) / n
    cm = [[0, 0, 0] for _ in LABELS]
    idx = {l: i for i, l in enumerate(LABELS)}
    for g, v in pairs:
        if g in idx and v in idx:
            cm[idx[g]][idx[v]] += 1
    per_class = {}
    f1s = []
    for i, lab in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(3)) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lab] = {"precision": round(prec, 4), "recall": round(rec, 4),
                          "f1": round(f1, 4)}
        f1s.append(f1)
    golds = [g for g, _ in pairs]
    preds_l = [v for _, v in pairs]
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(sum(f1s) / 3, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": LABELS,
        "cohens_kappa": cohen_kappa(golds, preds_l),
        "n": n,
    }

def pred_features(p: Dict[str, Any], tfidf: TfidfSpace) -> Dict[str, Any]:
    claim = (p.get("claim") or "").strip()
    rationale = (p.get("rationale") or "").strip() or (p.get("verdict") or "")
    docs = p.get("docs") or []
    doc_texts = [doc_text(d) for d in docs if doc_text(d).strip()]
    doc_ids = {d.get("doc_id") for d in docs if d.get("doc_id")}
    cited = {c for c in (p.get("cited_ids") or [])}
    support_sent = (p.get("support_sentence") or "").strip()
    contra_sent = (p.get("contra_sentence") or "").strip()
    verdict = p.get("verdict")

    rat_doc_cos = [tfidf.cosine(rationale, dt) for dt in doc_texts]
    rat_doc_jac = [jaccard(rationale, dt) for dt in doc_texts]
    claim_doc_cos = [tfidf.cosine(claim, dt) for dt in doc_texts]
    claim_doc_jac = [jaccard(claim, dt) for dt in doc_texts]

    if cited:
        cite_ground = len(cited & doc_ids) / len(cited)
    else:
        cite_ground = 0.0

    def sent_in_docs(sent: str) -> float:
        if not sent or not doc_texts:
            return 0.0
        st = set(content_tokens(sent))
        if not st:
            return 0.0
        best = 0.0
        for dt in doc_texts:
            dtoks = set(content_tokens(dt))
            if not dtoks:
                continue
            best = max(best, len(st & dtoks) / len(st))
        return best

    return {
        "claim": claim,
        "rationale": rationale,
        "verdict": verdict,
        "n_docs": len(doc_texts),
        "rat_doc_cos": rat_doc_cos,
        "rat_doc_jac": rat_doc_jac,
        "claim_doc_cos": claim_doc_cos,
        "claim_doc_jac": claim_doc_jac,
        "cite_ground": cite_ground,
        "support_grounding": sent_in_docs(support_sent),
        "contra_grounding": sent_in_docs(contra_sent),
        "doc_texts": doc_texts,
    }

def pillar2_row(f: Dict[str, Any], tfidf: TfidfSpace) -> Dict[str, Any]:
    rationale = f["rationale"]
    if f["n_docs"] == 0:
        faith = 0.0
    else:
        sent_scores = []
        for s in sentences(rationale) or [rationale]:
            best = max((token_f1(s, dt) for dt in f["doc_texts"]), default=0.0)
            sent_scores.append(min(1.0, best * 2.0))
        grounding = sum(sent_scores) / len(sent_scores) if sent_scores else 0.0
        faith = round(0.7 * grounding + 0.3 * f["cite_ground"], 4)
    ar = round(min(1.0, tfidf.cosine(rationale, f["claim"]) * 1.3), 4)
    return {"faithfulness": faith, "answer_relevancy": ar}

def judge_lexical(f: Dict[str, Any]) -> str:
    if f["n_docs"] == 0:
        return "abstain"
    g = max(f["rat_doc_cos"], default=0.0)
    if g >= 0.45:
        return "yes"
    if g < 0.18:
        return "no"
    return "abstain"

def judge_citation(f: Dict[str, Any], p: Dict[str, Any]) -> str:
    cited = p.get("cited_ids") or []
    verdict = f["verdict"]
    if verdict == "NEI":
        return "abstain" if not cited else ("yes" if f["cite_ground"] >= 0.5 else "no")
    if not cited:
        return "no"
    return "yes" if f["cite_ground"] >= 0.5 else "no"

def judge_structural(f: Dict[str, Any]) -> str:
    verdict = f["verdict"]
    max_rel = max(f["claim_doc_cos"], default=0.0)
    if f["n_docs"] == 0:
        return "abstain"
    if verdict == "SUPPORTED":
        if f["support_grounding"] >= 0.4 or max_rel >= 0.4:
            return "yes"
        if max_rel < 0.15:
            return "no"
        return "abstain"
    if verdict == "REFUTED":
        if f["contra_grounding"] >= 0.4 or max_rel >= 0.35:
            return "yes"
        if max_rel < 0.12:
            return "no"
        return "abstain"
    if max_rel < 0.2:
        return "yes"
    if max_rel >= 0.45:
        return "no"
    return "abstain"

def pillar3_system(preds: List[Dict[str, Any]], tfidf: TfidfSpace) -> Tuple[Dict, List[Dict]]:
    rows = []
    L, C, S = [], [], []
    for p in preds:
        f = pred_features(p, tfidf)
        jl, jc, js = judge_lexical(f), judge_citation(f, p), judge_structural(f)
        votes = [jl, jc, js]
        cnt = Counter(votes)
        majority = cnt.most_common(1)[0][0] if cnt.most_common(1)[0][1] >= 2 else "abstain"
        L.append(jl); C.append(jc); S.append(js)
        rows.append({
            "claim_id": p.get("claim_id"),
            "verdict": p.get("verdict"),
            "gold": p.get("gold_label"),
            "judges": {"lexical": jl, "citation": jc, "structural": js},
            "majority": majority,
        })
    n = len(rows)
    maj = [r["majority"] for r in rows]
    summary = {
        "n": n,
        "majority_yes_rate": round(maj.count("yes") / n, 4) if n else 0.0,
        "majority_no_rate": round(maj.count("no") / n, 4) if n else 0.0,
        "majority_abstain_rate": round(maj.count("abstain") / n, 4) if n else 0.0,
        "pairwise_kappa": {
            "lexical__citation": cohen_kappa(L, C),
            "lexical__structural": cohen_kappa(L, S),
            "citation__structural": cohen_kappa(C, S),
        },
        "per_judge_yes_rate": {
            "lexical": round(L.count("yes") / n, 4) if n else 0.0,
            "citation": round(C.count("yes") / n, 4) if n else 0.0,
            "structural": round(S.count("yes") / n, 4) if n else 0.0,
        },
    }
    kappas = [v for v in summary["pairwise_kappa"].values() if v is not None]
    summary["mean_pairwise_kappa"] = round(sum(kappas) / len(kappas), 4) if kappas else None
    return summary, rows

def framework_ragas(f: Dict[str, Any], tfidf: TfidfSpace) -> Dict[str, float]:
    faith = mean(f["rat_doc_cos"]) or 0.0
    ar = tfidf.cosine(f["rationale"], f["claim"])
    cc = f["claim_doc_cos"]
    if cc:
        num = sum(c / (i + 1) for i, c in enumerate(cc))
        den = sum(1 / (i + 1) for i in range(len(cc)))
        cp = num / den if den else 0.0
    else:
        cp = 0.0
    return {"faithfulness": round(min(1.0, faith * 1.6), 4),
            "answer_relevancy": round(min(1.0, ar * 1.3), 4),
            "context_precision": round(min(1.0, cp * 1.6), 4)}

def framework_deepeval(f: Dict[str, Any]) -> Dict[str, float]:
    sents = sentences(f["rationale"]) or [f["rationale"]]
    if f["n_docs"]:
        grounded = sum(1 for s in sents
                       if max((token_f1(s, dt) for dt in f["doc_texts"]), default=0.0) >= 0.22)
        faith = grounded / len(sents)
        ctx_rel = sum(1 for j in f["claim_doc_jac"] if j >= 0.07) / len(f["claim_doc_jac"])
    else:
        faith, ctx_rel = 0.0, 0.0
    ar = token_f1(f["rationale"], f["claim"])
    return {"faithfulness": round(faith, 4),
            "contextual_relevancy": round(ctx_rel, 4),
            "answer_relevancy": round(min(1.0, ar * 1.6), 4)}

def framework_openevals(f: Dict[str, Any]) -> Dict[str, float]:
    if f["n_docs"]:
        ground = 0.6 * max(f["rat_doc_jac"], default=0.0) + 0.4 * f["cite_ground"]
        retr = max(f["claim_doc_jac"], default=0.0)
    else:
        ground, retr = 0.0, 0.0
    return {"groundedness": round(min(1.0, ground * 2.2), 4),
            "retrieval_relevance": round(min(1.0, retr * 2.2), 4)}

def pillar4_system(preds: List[Dict[str, Any]], tfidf: TfidfSpace) -> Tuple[Dict, Dict[str, List]]:
    ragas_rows, deepeval_rows, openevals_rows = [], [], []
    for p in preds:
        f = pred_features(p, tfidf)
        cid = p.get("claim_id")
        rg = framework_ragas(f, tfidf); rg["claim_id"] = cid
        de = framework_deepeval(f); de["claim_id"] = cid
        oe = framework_openevals(f); oe["claim_id"] = cid
        ragas_rows.append(rg)
        deepeval_rows.append(de)
        openevals_rows.append(oe)

    def col(rows, key):
        return [r.get(key) for r in rows]

    summary = {
        "n_claims": len(preds),
        "ragas_means": {
            "faithfulness": mean(col(ragas_rows, "faithfulness")),
            "answer_relevancy": mean(col(ragas_rows, "answer_relevancy")),
            "context_precision": mean(col(ragas_rows, "context_precision")),
        },
        "deepeval_means": {
            "faithfulness": mean(col(deepeval_rows, "faithfulness")),
            "contextual_relevancy": mean(col(deepeval_rows, "contextual_relevancy")),
            "answer_relevancy": mean(col(deepeval_rows, "answer_relevancy")),
        },
        "openevals_means": {
            "groundedness": mean(col(openevals_rows, "groundedness")),
            "retrieval_relevance": mean(col(openevals_rows, "retrieval_relevance")),
        },
        "pearson_correlations": {
            "faithfulness__ragas_vs_deepeval":
                pearson(col(ragas_rows, "faithfulness"), col(deepeval_rows, "faithfulness")),
            "faithfulness__ragas_vs_openevals":
                pearson(col(ragas_rows, "faithfulness"), col(openevals_rows, "groundedness")),
            "faithfulness__deepeval_vs_openevals":
                pearson(col(deepeval_rows, "faithfulness"), col(openevals_rows, "groundedness")),
            "context__ragas_vs_deepeval":
                pearson(col(ragas_rows, "context_precision"), col(deepeval_rows, "contextual_relevancy")),
            "context__ragas_vs_openevals":
                pearson(col(ragas_rows, "context_precision"), col(openevals_rows, "retrieval_relevance")),
            "context__deepeval_vs_openevals":
                pearson(col(deepeval_rows, "contextual_relevancy"), col(openevals_rows, "retrieval_relevance")),
            "answer_relevancy__ragas_vs_deepeval":
                pearson(col(ragas_rows, "answer_relevancy"), col(deepeval_rows, "answer_relevancy")),
        },
    }
    rs = [v for v in summary["pearson_correlations"].values() if v is not None]
    summary["mean_pearson_r"] = round(sum(rs) / len(rs), 4) if rs else None
    return summary, {"ragas": ragas_rows, "deepeval": deepeval_rows, "openevals": openevals_rows}

def build_tfidf(exp: str, systems: List[str]) -> TfidfSpace:
    corpus = []
    for sysname in systems:
        for p in load_preds(exp, sysname):
            if p.get("claim"):
                corpus.append(p["claim"])
            if p.get("rationale"):
                corpus.append(p["rationale"])
            for d in (p.get("docs") or []):
                t = doc_text(d)
                if t.strip():
                    corpus.append(t)
    return TfidfSpace(corpus)

def run_experiment(exp: str, systems: List[str]) -> Dict[str, Any]:
    out_dir = EXP_DIR / exp / "pillars"
    out_dir.mkdir(parents=True, exist_ok=True)
    tfidf = build_tfidf(exp, systems)

    p1: Dict[str, Any] = {"per_system": {}}
    p2: Dict[str, Any] = {}
    p3: Dict[str, Any] = {"systems": {}, "judges": ["lexical", "citation", "structural"]}
    p4: Dict[str, Any] = {"systems": {}}

    for sysname in systems:
        preds = load_preds(exp, sysname)
        if not preds:
            print(f"  [{exp}/{sysname}] no predictions, skipped")
            continue

        p1["per_system"][sysname] = pillar1_system(preds)

        rows2 = []
        for p in preds:
            f = pred_features(p, tfidf)
            r = pillar2_row(f, tfidf)
            r["claim_id"] = p.get("claim_id")
            rows2.append(r)
        (out_dir / f"pillar2_rows_{sysname}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows2) + "\n")
        p2[sysname] = {
            "n": len(rows2),
            "faithfulness_mean": mean([r["faithfulness"] for r in rows2]),
            "answer_relevancy_mean": mean([r["answer_relevancy"] for r in rows2]),
        }

        s3, rows3 = pillar3_system(preds, tfidf)
        (out_dir / f"pillar3_rows_{sysname}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows3) + "\n")
        p3["systems"][sysname] = s3

        s4, rows4 = pillar4_system(preds, tfidf)
        for fw, rws in rows4.items():
            (out_dir / f"pillar4_{fw}_rows_{sysname}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rws) + "\n")
        p4["systems"][sysname] = s4

        print(f"  [{exp}/{sysname}] acc={p1['per_system'][sysname].get('accuracy')} "
              f"faith2={p2[sysname]['faithfulness_mean']} "
              f"p3_meanK={s3.get('mean_pairwise_kappa')} "
              f"p4_meanR={s4.get('mean_pearson_r')}")

    (out_dir / "pillar1.json").write_text(json.dumps(p1, indent=2))
    (out_dir / "pillar2.json").write_text(json.dumps(p2, indent=2))
    (out_dir / "pillar3.json").write_text(json.dumps(p3, indent=2))
    (out_dir / "pillar4.json").write_text(json.dumps(p4, indent=2))
    return {"pillar1": p1, "pillar2": p2, "pillar3": p3, "pillar4": p4}

def main():
    all_results = {}
    for exp, systems in EXPERIMENTS.items():
        print(f"=== {exp} ===")
        all_results[exp] = run_experiment(exp, systems)

    reliability = {"per_experiment": {}, "interpretation": {}}
    p3_kappas, p4_rs = [], []
    for exp, res in all_results.items():
        exp_block = {"pillar3_judge_kappa": {}, "pillar4_framework_pearson": {}}
        for sysname, s in res["pillar3"]["systems"].items():
            exp_block["pillar3_judge_kappa"][sysname] = s.get("mean_pairwise_kappa")
            if s.get("mean_pairwise_kappa") is not None:
                p3_kappas.append(s["mean_pairwise_kappa"])
        for sysname, s in res["pillar4"]["systems"].items():
            exp_block["pillar4_framework_pearson"][sysname] = s.get("mean_pearson_r")
            if s.get("mean_pearson_r") is not None:
                p4_rs.append(s["mean_pearson_r"])
        reliability["per_experiment"][exp] = exp_block

    reliability["overall"] = {
        "pillar3_mean_judge_kappa": round(sum(p3_kappas) / len(p3_kappas), 4) if p3_kappas else None,
        "pillar4_mean_framework_pearson_r": round(sum(p4_rs) / len(p4_rs), 4) if p4_rs else None,
    }
    reliability["interpretation"] = {
        "pillar3": "Mean pairwise Cohen's kappa between the three deterministic "
                   "judges (Lexical / Citation / Structural). Kappa > 0.4 = the "
                   "judges measure the same underlying 'verdict-is-grounded' "
                   "construct despite using independent rules.",
        "pillar4": "Mean Pearson r between the three deterministic frameworks "
                   "(Ragas / DeepEval / OpenEvals proxies) on shared constructs. "
                   "r > 0.7 = strong convergent validity; the RAG-triad scores "
                   "are not artefacts of one scoring formula.",
    }
    (COMBINED_DIR / "reliability_summary.json").write_text(json.dumps(reliability, indent=2))
    print("\n=== reliability_summary.json ===")
    print(json.dumps(reliability["overall"], indent=2))

if __name__ == "__main__":
    main()
