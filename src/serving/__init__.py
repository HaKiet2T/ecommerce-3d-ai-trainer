"""
Serving package for FastAPI 3D generation microservice.
"""

from .inference import CRMInferenceEngine, preprocess_product_image, synthesize_multiview_tensors
from .app import app

__all__ = ["app", "CRMInferenceEngine", "preprocess_product_image", "synthesize_multiview_tensors"]
