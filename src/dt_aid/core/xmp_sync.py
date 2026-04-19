from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import pyarrow.parquet as pq

from . import tagging, xmp
from .faces.embeddings import EmbeddingStore
from .state import cluster_label
from .tagging import Tag

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]


def _tags_for_image(
    image_rows_labels: list[tuple[str, int]],
    state_conn: sqlite3.Connection,
) -> list[Tag]:
    """
    Derive the managed tag set for a single image given each of its face
    rows as (label, cluster_id) tuples.

      label != ""                         -> people|<label>
      label == "" and cluster_id >= 0     -> promoted? people|<name> : people|unknown|cluster-NNN
      label == "" and cluster_id <  0     -> (no tag yet; unmatched/noise)

    The provenance tag is added if any face exists on the image.
    """
    if not image_rows_labels:
        return []
    tags: list[Tag] = []
    seen: set[str] = set()

    def _add(t: Tag) -> None:
        if t.value not in seen:
            seen.add(t.value)
            tags.append(t)

    for label, cluster_id in image_rows_labels:
        if label:
            _add(tagging.person_tag(label))
            continue
        if cluster_id is None or cluster_id < 0:
            continue
        promoted = cluster_label(state_conn, int(cluster_id))
        if promoted:
            _add(tagging.person_tag(promoted))
        else:
            _add(tagging.cluster_tag(int(cluster_id)))

    _add(tagging.faces_provenance_tag())
    return tags


def sync_xmp_for_images(
    image_paths: Iterable[str],
    *,
    store: EmbeddingStore,
    state_conn: sqlite3.Connection,
    progress: ProgressFn | None = None,
) -> int:
    """
    Rewrite the XMP sidecar for each given image so its managed tags
    reflect the current parquet state (matched labels, cluster IDs,
    promoted cluster labels). Non-managed tags on the XMP are preserved.

    Returns the number of XMPs written.
    """
    wanted = set(image_paths)
    if not wanted:
        return 0
    if not store.meta_path.exists():
        return 0

    table = pq.read_table(store.meta_path)
    per_image: dict[str, list[tuple[str, int]]] = defaultdict(list)
    paths_col = table["image_path"].to_pylist()
    labels_col = table["label"].to_pylist()
    clusters_col = table["cluster_id"].to_pylist()
    for p, lab, cid in zip(paths_col, labels_col, clusters_col):
        if p in wanted:
            per_image[p].append((lab or "", int(cid)))

    written = 0
    total = len(wanted)
    for idx, path_str in enumerate(sorted(wanted), start=1):
        if progress is not None:
            progress("xmp_sync", idx, total)
        rows_for_image = per_image.get(path_str, [])
        if not rows_for_image:
            continue
        image_path = Path(path_str)
        if not image_path.exists():
            log.debug("skip xmp sync (missing): %s", image_path)
            continue

        managed_tags = _tags_for_image(rows_for_image, state_conn)
        if not managed_tags:
            continue

        existing_flat, existing_hier = xmp.read_subjects(image_path)
        merged_hier = tagging.merge_managed(
            existing_hier if existing_hier else [], managed_tags
        )
        leaves = [xmp.leaf_label(t) for t in merged_hier]
        seen: set[str] = set()
        flat: list[str] = []
        for leaf in (*existing_flat, *leaves):
            if leaf not in seen:
                seen.add(leaf)
                flat.append(leaf)
        xmp.write_subjects(
            image_path, flat_tags=flat, hierarchical_tags=merged_hier
        )
        written += 1
    return written
