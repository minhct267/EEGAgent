"""Local Ollama embeddings for BGE-M3 (OpenAI-compatible /v1/embeddings)."""

from typing import List

import numpy as np

from llm_settings import embed_client, get_embed_settings

DEFAULT_BATCH_SIZE = 32


def _l2_normalize(vectors: List[List[float]]) -> List[List[float]]:
    if not vectors:
        return []
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (array / norms).tolist()


class BGEEmbedder:
    def __init__(self, model_name: str | None = None, batch_size: int = DEFAULT_BATCH_SIZE):
        settings = get_embed_settings()
        self.model_name = model_name or settings.model
        self.batch_size = batch_size
        self.client = embed_client()

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend(item.embedding for item in ordered)
        return _l2_normalize(embeddings)
