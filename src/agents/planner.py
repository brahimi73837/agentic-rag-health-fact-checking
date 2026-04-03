import logging
from typing import List
from src.llm import OllamaClient

log = logging.getLogger(__name__)

SYSTEM = "You rewrite health claims into PubMed search queries. Return JSON only."

PROMPT_TMPL = """Claim: "{claim}"

Task: Generate 3 diverse PubMed search queries to find evidence for or against this claim.
- Use biomedical terminology (disease names, drug names, mechanisms).
- Vary phrasing (specific terms, broader topic, outcome-focused).
- Each query 3-10 words. Short, keyword-style.

Return JSON exactly:
{{"queries": ["q1", "q2", "q3"]}}"""

def plan_queries(claim: str, llm: OllamaClient) -> List[str]:
    out = llm.json_call(PROMPT_TMPL.format(claim=claim), system=SYSTEM, max_tokens=1500)
    if out and isinstance(out.get("queries"), list):
        qs = [q.strip() for q in out["queries"] if isinstance(q, str) and q.strip()]
        if qs:
            return [claim] + qs[:3]
    log.warning("planner failed, falling back to claim-only")
    return [claim]
