"""Backward compatibility wrapper for src.pipeline.train"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.pipeline.train import main, train_crm

if __name__ == "__main__":
    main()
