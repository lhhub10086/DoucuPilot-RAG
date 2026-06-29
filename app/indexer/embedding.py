import hashlib
import math
import re
from abc import ABC, abstractmethod

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


class BaseEmbeddingProvider(ABC):
    provider_name: str
    model_name: str
    dimensions: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed one text string into a dense vector."""


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    provider_name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimensions = self.model.get_embedding_dimension()
        else:
            self.dimensions = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.astype(float).tolist()


class HashingEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic lightweight provider used only for tests or explicit fallback."""

    provider_name = "hashing"
    model_name = "hashing-vectorizer"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = self._tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        lowered = text.lower()
        tokens = TOKEN_RE.findall(lowered)
        if tokens:
            return tokens
        return [char for char in lowered if not char.isspace()]
