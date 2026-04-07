import logging
from typing import Dict, List, Tuple
from src.llm import OllamaClient
from src.config import MAX_EVIDENCE_CHARS

log = logging.getLogger(__name__)

def _ev_block(docs: List[Dict]) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        body = (d.get("abstract") or "")[:MAX_EVIDENCE_CHARS]
        lines.append(f"[{i}] id={d['doc_id']} | {d.get('title', '')}\n{body}")
    return "\n\n".join(lines) if lines else "(none)"

SUPPORT_SYS = "You check whether a specific assertion is directly stated in evidence. JSON only."
SUPPORT_TMPL = """Claim: "{claim}"

Evidence:
{ev}

Question: Does ANY abstract contain a sentence that directly asserts (not merely relates to) the claim's specific finding?
- Answer "yes" ONLY if you can copy a verbatim sentence that would be sufficient on its own to conclude the claim is true.
- Mechanism discussion, same-topic findings, or indirect implications → "no".

Return JSON: {{"direct_support": "yes|no", "sentence": "<verbatim quote if yes, else ''>"}}"""

def check_direct_support(claim: str, docs: List[Dict], llm: OllamaClient) -> Tuple[bool, str]:
    out = llm.json_call(
        SUPPORT_TMPL.format(claim=claim, ev=_ev_block(docs)),
        system=SUPPORT_SYS, max_tokens=200, temperature=0.0,
    )
    if not out:
        return False, ""
    ans = str(out.get("direct_support", "")).lower().strip()
    sent = str(out.get("sentence", ""))[:400]
    return (ans == "yes" and len(sent.strip()) > 10), sent

CONTRA_SYS = "You check whether evidence contradicts a claim. JSON only."
CONTRA_TMPL = """Claim: "{claim}"

Evidence:
{ev}

Question: Does ANY abstract contain a sentence that explicitly states the OPPOSITE of the claim, or a finding that directly contradicts it?
- "Opposite" = same subject, same measure, contradictory direction or negation.
- If the evidence is merely silent or unrelated → "no".
- If the evidence reports a null/negative finding for the claim's assertion → "yes".

Return JSON: {{"contradicts": "yes|no", "sentence": "<verbatim contradicting quote if yes, else ''>"}}"""

def check_contradiction(claim: str, docs: List[Dict], llm: OllamaClient) -> Tuple[bool, str]:
    out = llm.json_call(
        CONTRA_TMPL.format(claim=claim, ev=_ev_block(docs)),
        system=CONTRA_SYS, max_tokens=200, temperature=0.0,
    )
    if not out:
        return False, ""
    ans = str(out.get("contradicts", "")).lower().strip()
    sent = str(out.get("sentence", ""))[:400]
    return (ans == "yes" and len(sent.strip()) > 10), sent
