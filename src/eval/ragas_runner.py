import json
import logging
from pathlib import Path
from typing import List, Dict

log = logging.getLogger(__name__)

def _build_dataset(preds: List[Dict]):
    from datasets import Dataset
    rows = []
    for p in preds:
        if not p.get("docs"):
            continue
        contexts = [f"{d.get('title', '')}. {d.get('abstract', '')}" for d in p["docs"]]
        rows.append({
            "user_input": p["claim"],
            "response": p.get("rationale", "") or p.get("verdict", ""),
            "retrieved_contexts": contexts,
            "reference": p.get("gold_label", ""),
        })
    return Dataset.from_list(rows)

def run_ragas(preds: List[Dict], judge_model: str, out_path: Path,
              n_subset: int = 30) -> Dict:
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall, ResponseRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError as e:
        log.error("ragas missing: %s", e)
        return {"error": str(e)}

    preds_subset = [p for p in preds if p.get("docs")][:n_subset]
    if not preds_subset:
        return {"error": "no preds with contexts"}

    ds = _build_dataset(preds_subset)
    llm = LangchainLLMWrapper(ChatOllama(model=judge_model, temperature=0.0))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=judge_model))

    metrics = [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=emb),
               ContextPrecision(llm=llm)]
    try:
        result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb)
        df = result.to_pandas()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        summary = {m.name: float(df[m.name].mean()) for m in metrics if m.name in df.columns}
        summary["n"] = len(df)
        return summary
    except Exception as e:
        log.exception("ragas evaluate failed")
        return {"error": str(e)}
