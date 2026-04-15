import logging
import threading
import pytorch_lightning.utilities.argparse as _pla
if not hasattr(_pla, "_gpus_arg_default"):
    _pla._gpus_arg_default = lambda x: x
from typing import Dict, List, Optional
import torch
from torch import nn
from transformers import AutoTokenizer, LongformerModel

from src.config import MAX_EVIDENCE_CHARS

log = logging.getLogger(__name__)

LABEL_IDX = {0: "REFUTED", 1: "NEI", 2: "SUPPORTED"}
ENCODER_NAME = "allenai/longformer-large-4096"
ADDITIONAL_TOKENS = {
    "section_start": "<|sec|>", "section_end": "</|sec|>",
    "section_title_start": "<|sec-title|>", "section_title_end": "</|sec-title|>",
    "abstract_start": "<|abs|>", "abstract_end": "</|abs|>",
    "title_start": "<|title|>", "title_end": "</|title|>",
    "sentence_sep": "<|sent|>", "paragraph_sep": "<|par|>",
}

def _build_tokenizer():
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    tok.add_tokens(list(ADDITIONAL_TOKENS.values()))
    return tok

class _LabelClassifier(nn.Module):
    def __init__(self, hidden=1024, n_labels=3, dropout=0.1):
        super().__init__()
        self._linear_layers = nn.ModuleList([
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, n_labels),
        ])
        self._activations = nn.ModuleList([nn.GELU(), nn.Identity()])
        self._dropout = nn.ModuleList([nn.Dropout(dropout), nn.Dropout(0.0)])

    def forward(self, x):
        for lin, act, drop in zip(self._linear_layers, self._activations, self._dropout):
            x = drop(act(lin(x)))
        return x

class MultiVerSVerifier:
    _cache: Dict[str, "MultiVerSVerifier"] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, checkpoint_path: str) -> "MultiVerSVerifier":
        with cls._lock:
            if checkpoint_path not in cls._cache:
                cls._cache[checkpoint_path] = cls(checkpoint_path)
            return cls._cache[checkpoint_path]

    def __init__(self, checkpoint_path: str):
        self.tok = _build_tokenizer()
        self.encoder = LongformerModel.from_pretrained(ENCODER_NAME)
        self.encoder.resize_token_embeddings(len(self.tok))
        self.head = _LabelClassifier(self.encoder.config.hidden_size, 3,
                                     self.encoder.config.hidden_dropout_prob)
        self._load(checkpoint_path)
        self.encoder.eval(); self.head.eval()
        self.device = torch.device("cpu")
        self.encoder.to(self.device); self.head.to(self.device)

    def _load(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        sd = state.get("state_dict", state)
        enc_sd, head_sd = {}, {}
        for k, v in sd.items():
            if k.startswith("encoder."):
                enc_sd[k[len("encoder."):]] = v
            elif k.startswith("label_classifier."):
                head_sd[k[len("label_classifier."):]] = v
        cur = self.encoder.state_dict()
        if "embeddings.position_ids" not in enc_sd and "embeddings.position_ids" in cur:
            enc_sd["embeddings.position_ids"] = cur["embeddings.position_ids"]
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        log.info("encoder load: missing=%d unexpected=%d", len(missing), len(unexpected))
        if unexpected:
            log.debug("unexpected keys (first 5): %s", unexpected[:5])
        self.head.load_state_dict(head_sd, strict=True)

    @torch.no_grad()
    def predict_one(self, claim: str, doc: Dict) -> Dict:
        title = (doc.get("title") or "").strip()
        body = (doc.get("abstract") or "")[:MAX_EVIDENCE_CHARS]
        sents = [s.strip() for s in body.replace("\n", " ").split(". ") if s.strip()]
        cited_text = self.tok.eos_token.join(sents) if sents else ""
        if title:
            cited_text = title + self.tok.eos_token + cited_text
        text = claim + self.tok.eos_token + cited_text
        enc = self.tok(text, return_tensors="pt",
                       truncation=True, max_length=4096)
        ids = enc["input_ids"][0]
        is_special = (ids == self.tok.bos_token_id) | (ids == self.tok.eos_token_id)
        first_eos = (ids == self.tok.eos_token_id).nonzero()[0].item()
        is_claim = torch.arange(len(ids)) < first_eos
        gam = (is_special | is_claim).to(torch.long).unsqueeze(0)
        enc["global_attention_mask"] = gam
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.encoder(**enc)
        logits = self.head(out.pooler_output)
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        idx = int(torch.argmax(logits, dim=-1).item())
        return {
            "verdict": LABEL_IDX[idx],
            "probs": {LABEL_IDX[i]: float(probs[i]) for i in range(3)},
            "doc_id": doc.get("doc_id", ""),
        }

    def predict_aggregate(self, claim: str, docs: List[Dict],
                           label_threshold: Optional[float] = None) -> Dict:
        per_doc = [self.predict_one(claim, d) for d in docs]
        if not per_doc:
            return {"verdict": "NEI", "per_doc": [], "agg": "no_docs"}
        best_sup = max((p["probs"]["SUPPORTED"], p["doc_id"]) for p in per_doc)
        best_ref = max((p["probs"]["REFUTED"], p["doc_id"]) for p in per_doc)
        for p in per_doc:
            p["pred"] = max(p["probs"], key=p["probs"].get)
        non_nei_preds = [p for p in per_doc if p["pred"] != "NEI"]
        if not non_nei_preds:
            verdict = "NEI"; cited = []
        else:
            best = max(non_nei_preds,
                       key=lambda p: max(p["probs"]["SUPPORTED"], p["probs"]["REFUTED"]))
            verdict = best["pred"]
            cited = [best["doc_id"]]
        return {"verdict": verdict, "per_doc": per_doc, "cited_ids": cited,
                "best_supp_prob": best_sup[0], "best_ref_prob": best_ref[0]}
