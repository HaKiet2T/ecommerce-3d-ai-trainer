"""Backward compatibility wrapper for src.export.export_onnx"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.export.export_onnx import main, export_crm_to_onnx

if __name__ == "__main__":
    main()
