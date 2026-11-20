"""Offline two-stage semantic retrieval: bi-encoder top 20, cross-encoder top 5.

Both paths must refer to locally installed models.
"""
from __future__ import annotations
from pathlib import Path

class TwoStageRetriever:
    def __init__(self, encoder_path: str, reranker_path: str, device="cpu",
                 candidate_k=20, final_k=5, query_prefix=""):
        for role, path in (("Encoder", encoder_path), ("Reranker", reranker_path)):
            if not Path(path).is_dir():
                raise FileNotFoundError(f"{role} model directory is missing: {path}")
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from torch.nn import Identity

        self.encoder = SentenceTransformer(encoder_path, device=device, local_files_only=True)
        self.reranker = CrossEncoder(
            reranker_path, device=device, local_files_only=True, activation_fn=Identity(),
        )
        if self.reranker.model.config.num_labels != 1:
            raise ValueError("Reranker must produce one relevance logit per pair")
        self.query_prefix = query_prefix
        self.candidate_k = candidate_k
        self.final_k = final_k
        self._passages = None
        self._vectors = None

    def retrieve(self, query: str, passages: list[str], *, eligible_indices=None) -> list[tuple[int, float]]:
        if not passages:
            return []
        import numpy as np
        # Cache document embeddings across the twelve criterion queries.
        # Every case uses a fresh worker; no cross-case persistence.
        if self._passages is not passages and self._passages != passages:
            self._vectors = np.asarray(self.encoder.encode(
                passages, batch_size=32, normalize_embeddings=True, show_progress_bar=False))
            self._passages = list(passages)
        query_vector = np.asarray(self.encoder.encode(
            [self.query_prefix + query], batch_size=1, normalize_embeddings=True, show_progress_bar=False))[0]
        sims = self._vectors @ query_vector
        eligible = np.arange(len(passages)) if eligible_indices is None else np.asarray(eligible_indices, dtype=int)
        indices = eligible[np.argsort(-sims[eligible])[:self.candidate_k]].tolist()
        if not indices:
            return []
        pairs = [(query, passages[i]) for i in indices]
        scores = self.reranker.predict(pairs, batch_size=16)
        ranked = sorted(zip(indices, scores), key=lambda x: -float(x[1]))
        return [(int(i), float(1/(1+np.exp(-np.clip(float(score), -30, 30))))) for i, score in ranked[:self.final_k]]
