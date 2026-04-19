from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..config import Settings
from ..state import open_state
from ..xmp_sync import sync_xmp_for_images
from .embeddings import EmbeddingStore, ReferenceLibrary
from .matcher import FaceMatcher

log = logging.getLogger(__name__)


@dataclass
class RematchReport:
    candidates: int = 0
    new_matches: int = 0
    by_person: dict[str, int] = field(default_factory=dict)
    xmps_written: int = 0


def run_rematch(
    settings: Settings,
    *,
    threshold: float | None = None,
    write_xmp: bool = True,
    progress=None,
) -> RematchReport:
    """
    Rematch every unlabeled face embedding against the current reference
    library. Rows that exceed the threshold get `label = <name>` and
    `cluster_id = -2` in the parquet; affected XMPs are re-synced so the
    promoted tag replaces any stale `people|unknown|cluster-NNN`.

    Already-matched faces (label != "") are left alone. Re-running after
    adding a new reference picks up matches that earlier passes missed.
    """
    report = RematchReport()
    settings.ensure_dirs()

    refs = ReferenceLibrary(settings.face_references_dir).load_all()
    if not refs:
        log.warning("no references on disk — build some with `dt-aid faces build-refs`")
        return report

    store = EmbeddingStore(settings.face_embeddings_npy, settings.face_embeddings_meta)
    if not store.meta_path.exists() or not store.npy_path.exists():
        log.warning("no embedding store found; nothing to rematch")
        return report

    effective_threshold = (
        threshold if threshold is not None else settings.face_match_threshold
    )
    matcher = FaceMatcher(refs, threshold=effective_threshold)

    table = pq.read_table(store.meta_path)
    # Candidates: rows with empty label — covers both -1 noise and
    # unpromoted cluster rows. Already-matched rows (-2, label != "")
    # are excluded.
    candidates = table.filter(pc.equal(table["label"], ""))
    report.candidates = candidates.num_rows
    if candidates.num_rows == 0:
        log.info("no unlabeled rows to rematch")
        return report

    row_ids = np.asarray(candidates["row"].to_numpy(), dtype=np.int64)
    all_vecs = np.load(store.npy_path, mmap_mode="r")
    candidate_vecs = np.ascontiguousarray(all_vecs[row_ids])

    log.info(
        "rematching %d unlabeled faces against %d references (threshold=%.2f)",
        candidates.num_rows,
        len(refs),
        effective_threshold,
    )

    new_labels: dict[int, str] = {}
    new_clusters: dict[int, int] = {}
    by_person: dict[str, int] = {}
    for rid, vec in zip(row_ids, candidate_vecs):
        m = matcher.match(vec)
        if m is None:
            continue
        new_labels[int(rid)] = m.name
        new_clusters[int(rid)] = -2
        by_person[m.name] = by_person.get(m.name, 0) + 1

    report.new_matches = len(new_labels)
    report.by_person = by_person

    if not new_labels:
        log.info("no new matches found")
        return report

    store.update_assignments(labels=new_labels, cluster_ids=new_clusters)

    if not write_xmp:
        return report

    # Re-read to reflect the assignment write, then collect affected paths.
    refreshed = pq.read_table(store.meta_path)
    touched = set(new_labels.keys())
    paths: set[str] = set()
    for r, p in zip(
        refreshed["row"].to_pylist(), refreshed["image_path"].to_pylist()
    ):
        if int(r) in touched:
            paths.add(p)

    log.info("syncing XMP sidecars for %d images", len(paths))
    with open_state(settings.state_db) as state:
        report.xmps_written = sync_xmp_for_images(
            paths, store=store, state_conn=state, progress=progress
        )
    return report
