import os, json, time, threading
from pathlib import Path
from typing import Any, Optional

from src.config import RESULTS_DIR

TRACE_ROOT = RESULTS_DIR / "traces"
_LOCAL = threading.local()

class Tracer:
    def __init__(self, claim_id: str, run_tag: str):
        self.claim_id = claim_id
        self.run_tag = run_tag
        self.dir = TRACE_ROOT / run_tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{claim_id}.jsonl"
        self.start_t = time.time()
        self._fh = open(self.path, "w")
        self._seq = 0

    @classmethod
    def start(cls, claim_id: str, run_tag: str) -> "Tracer":
        t = cls(claim_id, run_tag)
        _LOCAL.tracer = t
        return t

    def emit(self, node: str, *, input: Any = None, output: Any = None,
             latency_ms: Optional[float] = None, meta: Optional[dict] = None):
        rec = {
            "claim_id": self.claim_id, "seq": self._seq, "node": node,
            "ts_rel_s": round(time.time() - self.start_t, 4),
            "latency_ms": latency_ms,
            "input": _safe(input), "output": _safe(output), "meta": meta or {},
        }
        self._seq += 1
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.close()
        if getattr(_LOCAL, "tracer", None) is self:
            _LOCAL.tracer = None

def active() -> Optional[Tracer]:
    return getattr(_LOCAL, "tracer", None)

def emit_if_active(node: str, **kw):
    t = active()
    if t is not None:
        t.emit(node, **kw)

def _safe(v: Any, max_str: int = 2000):
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v if len(v) <= max_str else v[:max_str] + f"…<+{len(v)-max_str}>"
    if isinstance(v, dict):
        return {k: _safe(x, max_str) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x, max_str) for x in v[:50]]
    try:
        return str(v)[:max_str]
    except Exception:
        return "<unrepr>"

def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)
