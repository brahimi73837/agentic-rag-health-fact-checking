import csv
import json
import random
import sys
from pathlib import Path
from collections import defaultdict

from src.config import (
    SCIFACT_DIR, HEALTHVER_DIR, EVAL_SPLITS_DIR,
    SCIFACT_LABEL_MAP, HEALTHVER_LABEL_MAP,
    EVAL_N_PER_CLASS, RANDOM_SEED,
)

csv.field_size_limit(sys.maxsize)
random.seed(RANDOM_SEED)

def _load_scifact_val():
    rows = []
    with open(SCIFACT_DIR / "claims_validation.csv", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            gold = SCIFACT_LABEL_MAP.get(row["evidence_label"], "NEI")
            rows.append({
                "claim_id": f"scifact_{row['id']}", "dataset": "scifact",
                "claim": row["claim"], "gold_label": gold,
                "cited_doc_ids": [x for x in (row.get("cited_doc_ids") or "").strip("[]").replace("'", "").split(",") if x.strip()],
            })
    return rows

def _load_healthver_test():
    rows = []
    with open(HEALTHVER_DIR / "healthver_test.csv", newline="") as f:
        r = csv.DictReader(f)
        seen_claims = set()
        for row in r:
            c = (row.get("claim") or "").strip()
            if not c or c in seen_claims:
                continue
            seen_claims.add(c)
            gold = HEALTHVER_LABEL_MAP.get(row["label"], "NEI")
            rows.append({
                "claim_id": f"healthver_{row['id']}", "dataset": "healthver",
                "claim": c, "gold_label": gold, "cited_doc_ids": [],
            })
    return rows

def _stratify(rows, n_per_class):
    by = defaultdict(list)
    for r in rows:
        by[r["gold_label"]].append(r)
    out = []
    for lab, items in by.items():
        random.shuffle(items)
        out.extend(items[:n_per_class])
    random.shuffle(out)
    return out

def build_splits():
    EVAL_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    sf = _stratify(_load_scifact_val(), EVAL_N_PER_CLASS)
    hv = _stratify(_load_healthver_test(), EVAL_N_PER_CLASS)

    def _write(path, rows):
        with open(path, "w") as w:
            for r in rows:
                w.write(json.dumps(r) + "\n")

    _write(EVAL_SPLITS_DIR / "scifact_eval.jsonl", sf)
    _write(EVAL_SPLITS_DIR / "healthver_eval.jsonl", hv)
    _write(EVAL_SPLITS_DIR / "combined_eval.jsonl", sf + hv)
    print(f"[splits] scifact={len(sf)} healthver={len(hv)} combined={len(sf) + len(hv)}")

if __name__ == "__main__":
    build_splits()
