import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CORPUS_JSONL
from src.llm import OllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("sdg")

DEICTIC = re.compile(
    r"\b(this study|the study|we |our |the patients?|the trial|the authors?|"
    r"shown (above|below)|in this paper|in this article|the present|"
    r"the (current|aforementioned|following)|herein|as (described|mentioned))\b",
    re.I,
)
HEALTH_GATE = re.compile(
    r"\b(patient|cancer|tumor|treatment|therapy|drug|disease|infection|virus|"
    r"vaccine|clinical|trial|symptom|diagnos|mortality|prognos|outcome|"
    r"randomi[sz]ed|placebo|cohort|incidence|prevalence|risk|mg|dose|therapy)",
    re.I,
)

GEN_SYSTEM = (
    "You generate health-claim triples for evidence-verification benchmarks. "
    "Return JSON only."
)
GEN_PROMPT = """Source abstract (doc_id={doc_id}):
\"\"\"{abstract}\"\"\"

Generate THREE health claims about this abstract for an evidence-verification benchmark.
Each claim must be a single self-contained declarative sentence, 8-22 words.
Forbidden: "this study", "the patients", "we", "our", "the trial", deictic references.

1. SUPPORTED — a specific factual finding the abstract DIRECTLY ASSERTS.
   Use named entities (specific drug, condition, mechanism, effect direction).
   The claim must be paraphrased, not copied verbatim.
2. REFUTED — take the SUPPORTED finding and flip it so the abstract DIRECTLY CONTRADICTS it.
   Concrete flips: increase↔decrease, higher↔lower, effective↔ineffective,
   associated↔not associated, significant↔non-significant, A causes B → A does not cause B.
   Keep the same entities; only invert the assertion direction or polarity.
   The result must be unambiguously falsified by the abstract text.
3. NEI — a topic-adjacent claim the abstract NEITHER asserts NOR refutes.
   Uses an entity mentioned in the abstract but asks something the abstract is silent on.

Return JSON exactly:
{{"supported": "...", "refuted": "...", "nei": "..."}}"""

VAL_SYSTEM = "You verify claims against scientific abstracts. JSON only."
VAL_PROMPT = """Abstract:
\"\"\"{abstract}\"\"\"

Claim: \"{claim}\"

Does the abstract:
- SUPPORTED: directly support / assert the claim?
- REFUTED: directly contradict / negate the claim?
- NEI: neither (insufficient information, off-topic, or only tangentially related)?

Return JSON: {{"label": "SUPPORTED|REFUTED|NEI", "evidence": "<one short sentence quoted from abstract or 'none'>"}}"""

def _norm_abstract(a):
    if isinstance(a, list): return " ".join(a)
    if isinstance(a, str):
        s = a.strip()
        if s.startswith("[") and ("'" in s or '"' in s):
            try:
                import ast; v = ast.literal_eval(s); return " ".join(v) if isinstance(v, list) else s
            except Exception: return s
        return s
    return ""

def quality_ok(claim: str) -> bool:
    if not claim:
        return False
    n_words = len(claim.split())
    if not (5 <= n_words <= 28):
        return False
    if DEICTIC.search(claim):
        return False
    if claim.count(".") > 1:
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-abstracts", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/sm_claims/synth_claims.jsonl")
    ap.add_argument("--rejects", default="data/sm_claims/synth_rejects.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out); rej_path = Path(args.rejects)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("loading corpus from %s", CORPUS_JSONL)
    pool = []
    with open(CORPUS_JSONL) as f:
        for line in f:
            try: pool.append(json.loads(line))
            except Exception: pass
    log.info("corpus pool: %d docs", len(pool))
    random.shuffle(pool)

    llm = OllamaClient()
    kept_fh = open(out_path, "w"); rej_fh = open(rej_path, "w")
    kept_n = 0; tried = 0; idx = 0
    target = args.n_abstracts

    for d in pool:
        if kept_n >= target * 3 or idx >= target * 6: break
        idx += 1
        doc_id = d.get("doc_id", "")
        abstract = _norm_abstract(d.get("abstract") or d.get("a") or "")
        title = d.get("title") or d.get("t") or ""
        n_w = len(abstract.split())
        if not (100 <= n_w <= 400): continue
        if not HEALTH_GATE.search(abstract): continue
        if "review" in title.lower() or "editorial" in title.lower(): continue
        tried += 1

        out = llm.json_call(
            GEN_PROMPT.format(doc_id=doc_id, abstract=abstract[:2400]),
            system=GEN_SYSTEM, max_tokens=350, temperature=0.3,
        )
        if not out: continue
        triples = []
        for label_key, gold in [("supported", "SUPPORTED"), ("refuted", "REFUTED"), ("nei", "NEI")]:
            claim = (out.get(label_key) or "").strip()
            if not quality_ok(claim):
                rej_fh.write(json.dumps({"doc_id": doc_id, "label": gold, "claim": claim,
                                         "reason": "quality"}) + "\n")
                continue
            v = llm.json_call(
                VAL_PROMPT.format(abstract=abstract[:2400], claim=claim),
                system=VAL_SYSTEM, max_tokens=200, temperature=0.0,
            )
            v_label = (v or {}).get("label", "").upper().strip() if v else ""
            if v_label != gold:
                rej_fh.write(json.dumps({"doc_id": doc_id, "label": gold, "claim": claim,
                                         "validator_label": v_label, "reason": "validator_disagree"}) + "\n")
                continue
            triples.append({
                "claim_id": f"syn_{kept_n:03d}",
                "claim": claim,
                "gold_label": gold,
                "cited_doc_id": doc_id,
                "cited_title": title,
                "validator_evidence": (v or {}).get("evidence", ""),
            })
            kept_n += 1
        for t in triples:
            kept_fh.write(json.dumps(t, ensure_ascii=False) + "\n"); kept_fh.flush()
        if tried % 5 == 0:
            log.info("tried=%d kept=%d (target=%d)", tried, kept_n, target * 3)

    kept_fh.close(); rej_fh.close()
    log.info("DONE tried=%d kept=%d", tried, kept_n)
    recs = [json.loads(l) for l in open(out_path)]
    from collections import Counter
    print("balance:", dict(Counter(r["gold_label"] for r in recs)))

if __name__ == "__main__":
    main()
