"""Backward compatibility wrapper for src.models"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.crm import ConvolutionalReconstructionModel, TriplaneSDFDecoder, MultiViewUNet

__all__ = ["ConvolutionalReconstructionModel", "TriplaneSDFDecoder", "MultiViewUNet"]
