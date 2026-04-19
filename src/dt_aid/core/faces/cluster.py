from __future__ import annotations

import numpy as np


def cluster_unknowns(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
) -> np.ndarray:
    """
    Cluster face embeddings with HDBSCAN. Returns an int array of cluster
    labels aligned to `embeddings`; -1 means noise.

    Assumes embeddings are L2-normalized. Euclidean distance on unit
    vectors is monotonically related to cosine distance
    (|a - b|^2 = 2 - 2 * a.b), so we use the euclidean metric with
    HDBSCAN's ball-tree implementation — O(N log N) memory instead of the
    O(N^2) blowup of a precomputed distance matrix. A 32k-embedding run
    costs roughly 70 MB instead of 7 GB.
    """
    if embeddings.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=max(2, min_cluster_size),
        min_samples=min_samples,
        allow_single_cluster=False,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(embeddings.astype(np.float64))
    return labels.astype(np.int64)


def build_references_from_faces(
    embeddings: np.ndarray, labels: list[str]
) -> dict[str, np.ndarray]:
    """Group embeddings by label (skipping empty labels) and return dict."""
    out: dict[str, list[np.ndarray]] = {}
    for emb, lab in zip(embeddings, labels):
        if not lab:
            continue
        out.setdefault(lab, []).append(emb)
    return {k: np.stack(v) for k, v in out.items()}
