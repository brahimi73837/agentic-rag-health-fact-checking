import logging
from typing import List, Dict, Optional, Tuple
from src.llm import OllamaClient
from src.config import LABELS, MAX_EVIDENCE_CHARS

log = logging.getLogger(__name__)

SYSTEM = "You verify health claims using scientific abstracts. Return JSON only."

PROMPT_TMPL = """Claim: "{claim}"

Evidence:
{evidence_block}

Task: Classify the claim as one of:
- SUPPORTED: evidence clearly supports the claim.
- REFUTED: evidence clearly contradicts the claim.
- NEI: evidence is insufficient, unrelated, or ambiguous.

Think step by step, then answer.

Return JSON exactly:
{{"verdict": "SUPPORTED|REFUTED|NEI", "rationale": "2-3 sentence explanation citing evidence", "cited_ids": ["id1", "id2"]}}"""

def _format_evidence(docs: List[Dict]) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        body = (d.get("abstract") or "")[:MAX_EVIDENCE_CHARS]
        title = d.get("title", "")
        lines.append(f"[{i}] id={d['doc_id']} | {title}\n{body}")
    return "\n\n".join(lines) if lines else "(no evidence retrieved)"

def verify(claim: str, docs: List[Dict], llm: OllamaClient) -> Dict:
    ev = _format_evidence(docs)
    prompt = PROMPT_TMPL.format(claim=claim, evidence_block=ev)
    out = llm.json_call(prompt, system=SYSTEM, max_tokens=350)
    if not out:
        return {"verdict": "NEI", "rationale": "parse_error", "cited_ids": [], "parse_error": True}
    verdict = str(out.get("verdict", "")).upper().strip()
    if verdict not in LABELS:
        verdict = "NEI"
    return {
        "verdict": verdict,
        "rationale": str(out.get("rationale", ""))[:800],
        "cited_ids": list(out.get("cited_ids") or []),
        "parse_error": False,
    }
