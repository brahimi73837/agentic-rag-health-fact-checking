import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR
from src.llm import OllamaClient
from src.trace import Tracer
from src.eval.run_eval import _run_pipeline, PER_CLAIM_TIMEOUT, _ClaimTimeout, _alarm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sm_eval")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="data/sm_claims/claims_final.jsonl")
    ap.add_argument("--pipelines", nargs="+",
                    default=["agentic", "standard_rag", "llm_only"])
    ap.add_argument("--limit", type=int, default=0, help="0=all, N=first N (smoke)")
    ap.add_argument("--config-name", default="sm_cn",
                    help="bucket name under results/predictions/<pipeline>/")
    ap.add_argument("--split-name", default="sm",
                    help="prefix of preds jsonl + traces folder")
    args = ap.parse_args()

    claims_path = Path(args.claims)
    claims = [json.loads(l) for l in open(claims_path)]
    if args.limit > 0: claims = claims[:args.limit]
    log.info("loaded %d claims from %s", len(claims), claims_path)

    llm = OllamaClient()

    for pipeline in args.pipelines:
        out_dir = RESULTS_DIR / "predictions" / pipeline / args.config_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.split_name}.jsonl"

        done_ids = set()
        if out_path.exists():
            for line in open(out_path):
                try: done_ids.add(json.loads(line)["claim_id"])
                except Exception: pass
            log.info("[%s] resume: %d done", pipeline, len(done_ids))

        run_tag = f"{pipeline}_{args.config_name}"
        with open(out_path, "a") as w:
            for i, c in enumerate(claims):
                cid = c["claim_id"]
                if cid in done_ids: continue
                t0 = time.time()
                tr = Tracer.start(claim_id=cid, run_tag=run_tag)
                try:
                    signal.signal(signal.SIGALRM, _alarm)
                    signal.alarm(PER_CLAIM_TIMEOUT)
                    try:
                        res = _run_pipeline(pipeline, c["claim"], llm, config={})
                    finally:
                        signal.alarm(0)
                except (Exception, _ClaimTimeout) as e:
                    signal.alarm(0)
                    log.exception("[%s] %s failed", pipeline, cid)
                    res = {"verdict": "NEI", "parse_error": True, "error": str(e),
                           "pipeline": pipeline, "docs": []}
                finally:
                    tr.close()
                res["claim_id"] = cid
                res["dataset"] = "sm_cn"
                res["gold_label"] = c["gold_label"]
                w.write(json.dumps(res) + "\n"); w.flush()
                log.info("[%s] %d/%d %s pred=%s gold=%s t=%.1fs",
                         pipeline, i+1, len(claims), cid,
                         res.get("verdict"), c["gold_label"], time.time()-t0)

if __name__ == "__main__":
    main()
