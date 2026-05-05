import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import classification_metrics, bootstrap_f1_ci
from src.eval.compare import pairwise

def parse_kv(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1); out[k] = Path(v)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="name=path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred_paths = parse_kv(args.preds)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {"per_system": {}, "pairwise_mcnemar": pairwise(pred_paths)}
    for name, path in pred_paths.items():
        yt, yp = [], []
        for line in open(path):
            r = json.loads(line)
            yt.append(r["gold_label"]); yp.append(r.get("verdict", "NEI"))
        m = classification_metrics(yt, yp)
        m["bootstrap_f1_ci"] = bootstrap_f1_ci(yt, yp)
        summary["per_system"][name] = m

    out_path.write_text(json.dumps(summary, indent=2, default=str))
    for name, m in summary["per_system"].items():
        print(f"{name:15s} acc={m['accuracy']:.3f} macro-F1={m['macro_f1']:.3f} "
              f"CI=[{m['bootstrap_f1_ci']['lo']:.3f},{m['bootstrap_f1_ci']['hi']:.3f}] n={m['n']}")
    print("--- McNemar ---")
    for k, v in summary["pairwise_mcnemar"].items():
        mc = v["mcnemar"]; print(f"{k}: b={mc['b']} c={mc['c']} p={mc['p_value']:.4f}")

if __name__ == "__main__":
    main()
