import json
from pathlib import Path
from typing import List, Dict, Tuple
from src.eval.metrics import mcnemar_test

def _load(path: Path) -> Tuple[List[str], List[str], List[str]]:
    ids, gold, pred = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["claim_id"])
            gold.append(r["gold_label"])
            pred.append(r.get("verdict", "NEI"))
    return ids, gold, pred

def align(a: Path, b: Path):
    ia, ga, pa = _load(a)
    ib, gb, pb = _load(b)
    map_a = dict(zip(ia, zip(ga, pa)))
    map_b = dict(zip(ib, zip(gb, pb)))
    common = [k for k in ia if k in map_b]
    yt = [map_a[k][0] for k in common]
    pa2 = [map_a[k][1] for k in common]
    pb2 = [map_b[k][1] for k in common]
    return yt, pa2, pb2, common

def pairwise(pred_paths: Dict[str, Path]) -> Dict:
    names = list(pred_paths.keys())
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            yt, pa, pb, common = align(pred_paths[a], pred_paths[b])
            out[f"{a}__vs__{b}"] = {
                "n_common": len(common),
                "mcnemar": mcnemar_test(yt, pa, pb),
            }
    return out
