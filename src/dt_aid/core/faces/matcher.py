from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Match:
    name: str
    similarity: float


class FaceMatcher:
    """
    Matches a query embedding against a library of per-person reference
    vectors using max cosine similarity. Assumes all vectors are
    L2-normalized.
    """

    def __init__(self, references: dict[str, np.ndarray], threshold: float = 0.5) -> None:
        self._names: list[str] = list(references.keys())
        self._threshold = threshold
        if not self._names:
            self._matrix = np.zeros((0, 512), dtype=np.float32)
            self._offsets = np.zeros((1,), dtype=np.int64)
        else:
            parts = [references[n].astype(np.float32) for n in self._names]
            self._matrix = np.concatenate(parts, axis=0)
            sizes = [p.shape[0] for p in parts]
            self._offsets = np.concatenate([[0], np.cumsum(sizes)])

    def match(self, embedding: np.ndarray) -> Match | None:
        if self._matrix.shape[0] == 0:
            return None
        q = embedding.astype(np.float32)
        sims = self._matrix @ q  # already unit-norm on both sides
        best_name: str | None = None
        best_sim = -1.0
        for i, name in enumerate(self._names):
            lo, hi = self._offsets[i], self._offsets[i + 1]
            s = float(sims[lo:hi].max())
            if s > best_sim:
                best_sim = s
                best_name = name
        if best_name is None or best_sim < self._threshold:
            return None
        return Match(name=best_name, similarity=best_sim)
