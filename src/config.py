import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_SPLITS_DIR = DATA_DIR / "eval_splits"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = ROOT / "index"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

EXT_DATASETS = ROOT.parent / "Thesis" / "Datasets"
SCIFACT_DIR = EXT_DATASETS / "SciFact"
HEALTHVER_DIR = EXT_DATASETS / "HealthVer"
PUBMEDQA_DIR = EXT_DATASETS / "PubMedQA"

CORPUS_JSONL = CORPUS_DIR / "corpus.jsonl"
DOC_IDS_NPY = INDEX_DIR / "doc_ids.npy"
FAISS_INDEX = INDEX_DIR / "faiss.index"
CORPUS_META = INDEX_DIR / "corpus_meta.jsonl"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)

EMBED_MODEL = os.getenv("EMBED_MODEL", "ncbi/MedCPT-Query-Encoder")
ARTICLE_EMBED_MODEL = os.getenv("ARTICLE_EMBED_MODEL", "ncbi/MedCPT-Article-Encoder")
CROSS_ENCODER = os.getenv("CROSS_ENCODER", "ncbi/MedCPT-Cross-Encoder")
EMBED_DIM = 768

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
NCBI_EMAIL = os.getenv("NCBI_EMAIL")
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

TOP_K_RETRIEVE = 30
TOP_K_RERANK = 5

PUBMED_INDEX_DIR = DATA_DIR / "pubmed_index"
PUBMED_FAISS = PUBMED_INDEX_DIR / "pubmed.faiss"
PUBMED_PMIDS = PUBMED_INDEX_DIR / "pmids.npy"
PUBMED_DB = PUBMED_INDEX_DIR / "pubmed.db"
USE_PUBMED_POOL = os.getenv("USE_PUBMED_POOL", "0") == "1"
PUBMED_NPROBE = int(os.getenv("PUBMED_NPROBE", "32"))
PUBMED_TOP_K = int(os.getenv("PUBMED_TOP_K", "30"))
LIVE_FALLBACK_SCORE_THRESHOLD = 0.35
MAX_EVIDENCE_CHARS = 1500

MAX_CRITIC_RETRIES = 1
GROUNDEDNESS_ROUGE_THRESHOLD = 0.15

LABELS = ["SUPPORTED", "REFUTED", "NEI"]
SCIFACT_LABEL_MAP = {"SUPPORT": "SUPPORTED", "CONTRADICT": "REFUTED", "": "NEI"}
HEALTHVER_LABEL_MAP = {"Supports": "SUPPORTED", "Refutes": "REFUTED", "Neutral": "NEI"}

EVAL_N_PER_CLASS = 15
BOOTSTRAP_ITER = 10000
RANDOM_SEED = 42
