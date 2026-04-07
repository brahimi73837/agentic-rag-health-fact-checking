import logging
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer

from src.llm import OllamaClient
from src.config import GROUNDEDNESS_ROUGE_THRESHOLD

log = logging.getLogger(__name__)
_SCORER = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

SYSTEM = "You refine failed searches. Return JSON only."

RETRY_PROMPT = """Claim: "{claim}"
Previous queries: {prev_queries}
Previous verdict: {verdict} — but evidence was weak or unrelated.

Task: Propose 2 NEW, different queries likely to surface stronger evidence.
- Try alternative terms (mechanism, population, outcome).
- Avoid repeating prior queries.

Return JSON:
{{"queries": ["q1", "q2"]}}"""

def groundedness_score(rationale: str, docs: List[Dict], cited_ids: List[str]) -> float:
    if not rationale:
        return 0.0
    if cited_ids:
        texts = [d["abstract"] for d in docs if d["doc_id"] in set(cited_ids)]
    else:
        texts = [d["abstract"] for d in docs]
    if not texts:
        return 0.0
    evidence = " ".join(texts)
    return _SCORER.score(evidence, rationale)["rouge1"].fmeasure

def needs_retry(verdict_dict: Dict, docs: List[Dict]) -> Tuple[bool, float]:
    if verdict_dict.get("parse_error"):
        return True, 0.0
    g = groundedness_score(verdict_dict.get("rationale", ""), docs, verdict_dict.get("cited_ids", []))
    return g < GROUNDEDNESS_ROUGE_THRESHOLD, g

def refine_queries(claim: str, prev_queries: List[str], prev_verdict: str, llm: OllamaClient) -> List[str]:
    out = llm.json_call(
        RETRY_PROMPT.format(claim=claim, prev_queries=prev_queries, verdict=prev_verdict),
        system=SYSTEM, max_tokens=180,
    )
    if out and isinstance(out.get("queries"), list):
        qs = [q.strip() for q in out["queries"] if isinstance(q, str) and q.strip()]
        return qs[:2]
    return []
