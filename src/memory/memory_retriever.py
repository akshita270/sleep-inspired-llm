"""Find the stored facts most related to a question.

How it works: an embedding model turns each sentence into a list of numbers
(a vector) so that sentences with similar MEANING get similar vectors.
Similarity is the cosine of the angle between two vectors (1 = same
direction, 0 = unrelated). We embed the facts once, then for each question we
embed it and pick the top-k most similar facts.

The embedder runs on the CPU on purpose, so it never competes with the
language model for the Apple GPU's memory.
"""
import numpy as np
from sentence_transformers import SentenceTransformer


class MemoryRetriever:
    def __init__(self, model_name: str, top_k: int, min_similarity: float):
        self.embedder = SentenceTransformer(model_name, device="cpu")
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.items: list[dict] = []
        self.vectors: np.ndarray | None = None

    def index(self, items: list[dict]) -> None:
        """items: [{'experience_id': ..., 'statement': ...}, ...]"""
        self.items = items
        if items:
            self.vectors = self.embedder.encode(
                [i["statement"] for i in items], normalize_embeddings=True)
        else:
            self.vectors = None

    def retrieve(self, query: str) -> list[dict]:
        """Top-k facts with similarity >= min_similarity, best first."""
        if self.vectors is None:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.vectors @ q          # vectors are unit length, so this is cosine
        order = np.argsort(-scores)[: self.top_k]
        return [{**self.items[i], "score": float(scores[i])}
                for i in order if scores[i] >= self.min_similarity]
