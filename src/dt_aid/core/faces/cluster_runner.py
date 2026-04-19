from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..config import Settings
from ..state import open_state, upsert_cluster
from ..xmp_sync import sync_xmp_for_images
from .cluster import cluster_unknowns
from .embeddings import EmbeddingStore

log = logging.getLogger(__name__)


@dataclass
class ClusterReport:
    unmatched_total: int = 0
    clustered: int = 0         # rows assigned cluster_id >= 0
    noise: int = 0             # rows that stayed at -1
    new_clusters: int = 0      # distinct cluster IDs produced
    xmps_written: int = 0


def run_cluster(
    settings: Settings,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    write_xmp: bool = True,
    progress=None,
) -> ClusterReport:
    """
    Cluster every face with cluster_id == -1, write new cluster IDs back
    into the parquet + state.db, and optionally re-sync XMPs for affected
    images. Matched faces (cluster_id == -2) are untouched.
    """
    report = ClusterReport()
    settings.ensure_dirs()

    store = EmbeddingStore(settings.face_embeddings_npy, settings.face_embeddings_meta)
    if not store.meta_path.exists() or not store.npy_path.exists():
        log.warning("no embedding store found; nothing to cluster")
        return report

    table = pq.read_table(store.meta_path)
    unmatched_mask = pc.equal(table["cluster_id"], -1)
    unmatched = table.filter(unmatched_mask)
    report.unmatched_total = unmatched.num_rows
    if unmatched.num_rows == 0:
        log.info("no unmatched faces to cluster")
        return report

    unmatched_row_ids = np.asarray(unmatched["row"].to_numpy(), dtype=np.int64)

    # Load embeddings only for the unmatched rows (memmap + fancy index).
    all_vecs = np.load(store.npy_path, mmap_mode="r")
    vecs = np.ascontiguousarray(all_vecs[unmatched_row_ids])

    log.info(
        "clustering %d unmatched faces (min_cluster_size=%d)",
        vecs.shape[0],
        min_cluster_size,
    )
    labels = cluster_unknowns(
        vecs,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    assert labels.shape[0] == unmatched_row_ids.shape[0]

    # Build row_id -> new_cluster_id mapping for rows that were assigned
    # a real cluster (>= 0). Noise stays at -1 (unchanged).
    assignments: dict[int, int] = {}
    for row_id, lab in zip(unmatched_row_ids, labels):
        if lab >= 0:
            assignments[int(row_id)] = int(lab)

    report.clustered = len(assignments)
    report.noise = int(vecs.shape[0] - len(assignments))
    report.new_clusters = int(len(set(assignments.values())))

    if not assignments:
        log.info("HDBSCAN produced no clusters at min_cluster_size=%d", min_cluster_size)
        return report

    store.update_assignments(cluster_ids=assignments)
    with open_state(settings.state_db) as state:
        for cid in sorted(set(assignments.values())):
            upsert_cluster(state, cid)

    if write_xmp:
        # Find the images touched by these row assignments.
        # Re-read meta (store.update_assignments just rewrote it).
        refreshed = pq.read_table(store.meta_path)
        touched_row_ids = set(assignments.keys())
        paths: set[str] = set()
        paths_col = refreshed["image_path"].to_pylist()
        rows_col = refreshed["row"].to_pylist()
        for r, p in zip(rows_col, paths_col):
            if int(r) in touched_row_ids:
                paths.add(p)
        log.info("syncing XMP sidecars for %d images", len(paths))
        with open_state(settings.state_db) as state:
            report.xmps_written = sync_xmp_for_images(
                paths, store=store, state_conn=state, progress=progress
            )

    return report
