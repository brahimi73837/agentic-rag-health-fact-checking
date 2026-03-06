# ThesisV3 — Agentic RAG for Health-Claim Verification

Reproducible prototype for the dissertation: a multi-agent RAG pipeline over
PubMed / SciFact / HealthVer that classifies health claims as
SUPPORTED / REFUTED / NEI with evidence groundedness.

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then fill in NCBI_API_KEY etc.
ollama pull qwen2.5:7b-instruct-q4_K_M
```

## Build (one-time)

```bash
.venv/bin/python scripts/01_build_corpus.py   # unified corpus JSONL
.venv/bin/python scripts/02_build_index.py    # MedCPT embeddings + FAISS IVF index
.venv/bin/python scripts/03_build_splits.py   # stratified eval splits
.venv/bin/python scripts/synth_claims_gen.py  # Experiment 3 synthetic claims
```

## Evaluate

```bash
.venv/bin/python scripts/run_sm_eval.py                  # Experiment 2 (social-media / Community Notes)
.venv/bin/python scripts/pillar1_verdict_acc.py          # accuracy, macro-F1, kappa
.venv/bin/python scripts/pillar2_fast.py                 # faithfulness + answer-relevancy
.venv/bin/python scripts/pillar3_judge_triangulation.py  # three-judge groundedness
.venv/bin/python scripts/pillar4_framework_xcheck.py     # cross-framework convergence
.venv/bin/python scripts/build_eval_pillars.py           # collate eval inputs
.venv/bin/python scripts/compute_eval_pillars.py         # collate eval outputs
```

Each script's exact flags are documented in its header. Prediction logs land in `results/`.

## Layout

- `src/config.py` — constants, paths, env loading
- `src/llm.py` — Ollama client (JSON mode + retry)
- `src/embeddings.py` — MedCPT Query / Article / Cross encoders
- `src/ncbi.py`, `src/pubmed_retrieval.py` — live PubMed via E-utilities (cached, rate-limited)
- `src/corpus_build.py`, `src/index_build.py` — one-time corpus / index builds
- `src/retrieval.py` — FAISS search + cross-encoder rerank + live fallback
- `src/agents/{planner,verifier,posthoc_checks,multivers_verifier}.py` — agent nodes
- `src/pipelines/{agentic,standard_rag,llm_only,agentic_multivers}.py` — the evaluated systems
- `src/eval/{splits,metrics,ragas_runner,run_eval,compare}.py` — eval harness
