import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from src.config import EMBED_MODEL, ARTICLE_EMBED_MODEL, CROSS_ENCODER

_QUERY_CACHE = {}
_ARTICLE_CACHE = {}
_CROSS_CACHE = {}

def _device() -> str:
    import os
    if os.environ.get("FORCE_MPS") == "1" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _load(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModel.from_pretrained(name).to(_device())
    m.eval()
    return tok, m

def get_query_encoder():
    if "m" not in _QUERY_CACHE:
        tok, m = _load(EMBED_MODEL)
        _QUERY_CACHE["tok"], _QUERY_CACHE["m"] = tok, m
    return _QUERY_CACHE["tok"], _QUERY_CACHE["m"]

def get_article_encoder():
    if "m" not in _ARTICLE_CACHE:
        tok, m = _load(ARTICLE_EMBED_MODEL)
        _ARTICLE_CACHE["tok"], _ARTICLE_CACHE["m"] = tok, m
    return _ARTICLE_CACHE["tok"], _ARTICLE_CACHE["m"]

def get_cross_encoder():
    if "m" not in _CROSS_CACHE:
        from transformers import AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(CROSS_ENCODER)
        m = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER).to(_device())
        m.eval()
        _CROSS_CACHE["tok"], _CROSS_CACHE["m"] = tok, m
    return _CROSS_CACHE["tok"], _CROSS_CACHE["m"]

@torch.no_grad()
def encode_queries(queries: List[str], batch_size: int = 16, max_len: int = 64) -> np.ndarray:
    tok, m = get_query_encoder()
    out = []
    for i in range(0, len(queries), batch_size):
        chunk = queries[i:i + batch_size]
        enc = tok(chunk, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(m.device)
        emb = m(**enc).last_hidden_state[:, 0, :]
        out.append(emb.cpu().numpy().astype("float32"))
    return np.vstack(out) if out else np.zeros((0, 768), dtype="float32")

@torch.no_grad()
def encode_articles(titles: List[str], abstracts: List[str], batch_size: int = 8, max_len: int = 512) -> np.ndarray:
    tok, m = get_article_encoder()
    out = []
    pairs = [[t or "", a or ""] for t, a in zip(titles, abstracts)]
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        enc = tok(chunk, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(m.device)
        emb = m(**enc).last_hidden_state[:, 0, :]
        out.append(emb.cpu().numpy().astype("float32"))
    return np.vstack(out) if out else np.zeros((0, 768), dtype="float32")

@torch.no_grad()
def rerank_cross(query: str, docs: List[str], batch_size: int = 4, max_len: int = 384) -> List[float]:
    tok, m = get_cross_encoder()
    pairs = [[query, d] for d in docs]
    scores = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        enc = tok(chunk, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(m.device)
        logits = m(**enc).logits.squeeze(-1)
        scores.extend(logits.cpu().numpy().tolist())
    return scores
