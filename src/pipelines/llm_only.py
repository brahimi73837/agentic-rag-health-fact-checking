import time
from typing import Dict
from src.llm import OllamaClient
from src.config import LABELS

SYSTEM = "You classify health claims using your knowledge. Return JSON only."
PROMPT = """Claim: "{claim}"

Classify as:
- SUPPORTED: supported by scientific consensus.
- REFUTED: contradicted by scientific consensus.
- NEI: insufficient or uncertain evidence.

Return JSON:
{{"verdict": "SUPPORTED|REFUTED|NEI", "rationale": "1-2 sentence explanation"}}"""

def run_llm_only(claim: str, llm: OllamaClient) -> Dict:
    t0 = time.time()
    out = llm.json_call(PROMPT.format(claim=claim), system=SYSTEM, max_tokens=200)
    if not out:
        return {"claim": claim, "verdict": "NEI", "rationale": "parse_error",
                "parse_error": True, "time_s": time.time() - t0, "pipeline": "llm_only",
                "cited_ids": [], "docs": [], "queries": []}
    verdict = str(out.get("verdict", "")).upper().strip()
    if verdict not in LABELS:
        verdict = "NEI"
    return {
        "claim": claim, "verdict": verdict, "rationale": str(out.get("rationale", ""))[:800],
        "cited_ids": [], "docs": [], "queries": [],
        "parse_error": False, "time_s": time.time() - t0, "pipeline": "llm_only",
    }
