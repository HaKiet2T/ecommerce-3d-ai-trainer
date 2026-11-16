"""Backward compatibility wrapper for src.serving.app"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.serving.app import app, main

if __name__ == "__main__":
    main()
