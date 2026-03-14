import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corpus_build import build_corpus

if __name__ == "__main__":
    build_corpus()
