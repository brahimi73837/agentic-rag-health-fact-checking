import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.splits import build_splits

if __name__ == "__main__":
    build_splits()
