import json
import signal
import logging
from pathlib import Path
from typing import List, Dict, Callable, Optional
from tqdm import tqdm

import os
PER_CLAIM_TIMEOUT = int(os.environ.get("PER_CLAIM_TIMEOUT", "300"))

class _ClaimTimeout(BaseException):
    pass

def _alarm(signum, frame):
    raise _ClaimTimeout(f"claim exceeded {PER_CLAIM_TIMEOUT}s")

from src.config import RESULTS_DIR, EVAL_SPLITS_DIR
from src.llm import OllamaClient
from src.pipelines.agentic import run_agentic
from src.pipelines.agentic_multivers import run_agentic_multivers
from src.pipelines.standard_rag import run_standard_rag
from src.pipelines.llm_only import run_llm_only
from src.eval.metrics import classification_metrics, bootstrap_f1_ci

log = logging.getLogger(__name__)

def load_split(name: str) -> List[Dict]:
    path = EVAL_SPLITS_DIR / f"{name}_eval.jsonl"
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out

def _run_pipeline(pipeline: str, claim: str, llm: OllamaClient, config: Dict) -> Dict:
    if pipeline == "agentic":
        return run_agentic(claim, llm, config=config)
    if pipeline == "agentic_multivers":
        return run_agentic_multivers(claim, llm, config=config)
    if pipeline == "standard_rag":
        return run_standard_rag(claim, llm)
    if pipeline == "llm_only":
        return run_llm_only(claim, llm)
    raise ValueError(pipeline)

def run_config(pipeline: str, split: str, config_name: str,
               config: Optional[Dict] = None, resume: bool = True) -> Path:
    out_dir = RESULTS_DIR / "predictions" / pipeline / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"

    done_ids = set()
    if resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["claim_id"])
                except Exception:
                    pass

    claims = load_split(split)
    llm = OllamaClient()
    config = config or {}

    with open(out_path, "a") as w:
        for c in tqdm(claims, desc=f"{pipeline}/{config_name}/{split}"):
            if c["claim_id"] in done_ids:
                continue
            try:
                signal.signal(signal.SIGALRM, _alarm)
                signal.alarm(PER_CLAIM_TIMEOUT)
                try:
                    res = _run_pipeline(pipeline, c["claim"], llm, config)
                finally:
                    signal.alarm(0)
            except (Exception, _ClaimTimeout) as e:
                signal.alarm(0)
                log.exception("pipeline fail: %s", c["claim_id"])
                res = {"verdict": "NEI", "parse_error": True, "error": str(e),
                       "pipeline": pipeline, "docs": []}
            res["claim_id"] = c["claim_id"]
            res["dataset"] = c["dataset"]
            res["gold_label"] = c["gold_label"]
            w.write(json.dumps(res) + "\n")
            w.flush()
    return out_path

def compute_metrics_file(pred_path: Path, metrics_path: Path) -> Dict:
    y_true, y_pred = [], []
    with open(pred_path) as f:
        for line in f:
            r = json.loads(line)
            y_true.append(r["gold_label"])
            y_pred.append(r.get("verdict", "NEI"))
    m = classification_metrics(y_true, y_pred)
    m["bootstrap_f1_ci"] = bootstrap_f1_ci(y_true, y_pred)
    m["predictions_path"] = str(pred_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(m, indent=2))
    return m
