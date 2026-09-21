from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


@dataclass
class HashingEmbedder:
    dimensions: int = 256

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(text) for text in texts]


@dataclass
class SentenceTransformerEmbedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required for SentenceTransformerEmbedder.") from exc
        self._model = SentenceTransformer(self.model_name)

    def encode(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [row.tolist() for row in vectors]
