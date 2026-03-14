import csv
import json
import logging
import random
import sys
from pathlib import Path

from src.config import (
    CORPUS_JSONL, SCIFACT_DIR, PUBMEDQA_DIR, HEALTHVER_DIR, RANDOM_SEED
)

PUBMEDQA_MAX = 15000
random.seed(RANDOM_SEED)

log = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)

def _iter_scifact_corpus():
    path = SCIFACT_DIR / "corpus_train.csv"
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            doc_id = f"scifact_{row.get('doc_id') or row.get('id', '')}"
            title = row.get("title", "")
            abstract = row.get("abstract", "")
            if not abstract:
                continue
            yield {"doc_id": doc_id, "source": "scifact",
                   "title": title, "abstract": abstract}

def _iter_pubmedqa():
    path = PUBMEDQA_DIR / "pqal" / "ori_pqau.json"
    with open(path) as f:
        data = json.load(f)
    keys = list(data.keys())
    if len(keys) > PUBMEDQA_MAX:
        keys = random.sample(keys, PUBMEDQA_MAX)
    for pmid in keys:
        rec = data[pmid]
        ctx = rec.get("CONTEXTS") or []
        labels = rec.get("LABELS") or []
        title = rec.get("QUESTION", "")
        if labels and ctx and len(labels) == len(ctx):
            pieces = [f"{lab}: {c}" for lab, c in zip(labels, ctx)]
            abstract = " ".join(pieces)
        else:
            abstract = " ".join(ctx)
        if not abstract.strip():
            continue
        yield {"doc_id": f"pubmedqa_{pmid}", "source": "pubmedqa",
               "title": title, "abstract": abstract}

def _iter_healthver():
    seen = set()
    for split in ["healthver_train.csv", "healthver_dev.csv", "healthver_test.csv"]:
        path = HEALTHVER_DIR / split
        if not path.exists():
            continue
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                ev = (row.get("evidence") or "").strip()
                if not ev or ev in seen:
                    continue
                seen.add(ev)
                row_id = row.get("id", "") or f"hv{len(seen)}"
                yield {"doc_id": f"healthver_{row_id}", "source": "healthver",
                       "title": row.get("question", "")[:200], "abstract": ev}

def build_corpus():
    CORPUS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    counts = {"scifact": 0, "pubmedqa": 0, "healthver": 0}
    with open(CORPUS_JSONL, "w") as w:
        for rec in _iter_scifact_corpus():
            w.write(json.dumps(rec) + "\n"); counts["scifact"] += 1
        for rec in _iter_pubmedqa():
            w.write(json.dumps(rec) + "\n"); counts["pubmedqa"] += 1
        for rec in _iter_healthver():
            w.write(json.dumps(rec) + "\n"); counts["healthver"] += 1
    print(f"[corpus] wrote {CORPUS_JSONL} counts={counts} total={sum(counts.values())}")
    return counts

if __name__ == "__main__":
    build_corpus()
