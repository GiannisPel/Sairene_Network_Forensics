import os
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = None

def _model_name() -> str:
    return os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_model_name())
    return _MODEL

def embed_text(text: str) -> np.ndarray:
    v = get_model().encode(text, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)
