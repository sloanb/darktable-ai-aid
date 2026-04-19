from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..config import Settings
from ..state import open_state
from ..xmp_sync import sync_xmp_for_images
from .detector import FaceDetector
from .embeddings import EmbeddingStore, ReferenceLibrary

log = logging.getLogger(__name__)


@dataclass
class AddImageReport:
    image_path: str
    faces_detected: int
    chosen_face_index: int
    reference_count_after: int  # vectors in <name>.npy after append
    parquet_row_updated: bool   # True if an existing row was relabeled
    xmp_written: bool


def _pick_face_index(faces, face_index: int | None) -> int:
    if not faces:
        raise ValueError("no faces detected in image")
    if face_index is not None:
        if face_index < 0 or face_index >= len(faces):
            raise IndexError(
                f"--face-index {face_index} out of range (image has {len(faces)} faces)"
            )
        return face_index
    # largest face by bbox area
    areas = [
        (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces
    ]
    return int(np.argmax(areas))


def _find_matching_parquet_row(
    image_path: str,
    query_embedding: np.ndarray,
    store: EmbeddingStore,
    *,
    min_similarity: float = 0.95,
) -> int | None:
    """
    If the image already has rows in the embedding store, find the row
    whose cached embedding is closest to `query_embedding`. Returns the
    `row` index if similarity exceeds `min_similarity`, else None.

    The high threshold exploits the fact that re-detecting the same face
    on the same image produces a near-identical embedding (same model,
    same input) — we expect similarities > 0.99 for true matches.
    """
    if not store.meta_path.exists() or not store.npy_path.exists():
        return None
    table = pq.read_table(store.meta_path)
    mask = pc.equal(table["image_path"], image_path)
    rows_for_image = table.filter(mask)
    if rows_for_image.num_rows == 0:
        return None
    row_ids = np.asarray(rows_for_image["row"].to_numpy(), dtype=np.int64)
    all_vecs = np.load(store.npy_path, mmap_mode="r")
    cached = np.ascontiguousarray(all_vecs[row_ids])
    sims = cached @ query_embedding.astype(np.float32)
    best = int(np.argmax(sims))
    if float(sims[best]) < min_similarity:
        return None
    return int(row_ids[best])


def run_add_image(
    settings: Settings,
    *,
    image_path: Path,
    name: str,
    face_index: int | None = None,
    providers: list[str] | None = None,
) -> AddImageReport:
    """
    Teach dt-aid that a specific face in a specific image is `name`.

      1. Detect faces on the image.
      2. Select the target face (by --face-index or largest).
      3. Append the face embedding to references/<name>.npy.
      4. If the image is already in the embedding store, update the
         matching face row to label=name, cluster_id=-2 and re-sync its
         XMP sidecar so `people|<name>` replaces any stale tag.

    This does NOT rematch the rest of the library — run `dt-aid faces
    rematch` afterward to pick up other instances of <name> now that the
    reference is richer.
    """
    settings.ensure_dirs()

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    detector = FaceDetector(
        models_dir=settings.models_dir,
        det_size=settings.face_det_size,
        det_score_threshold=settings.face_det_score_threshold,
        providers=providers,
    )
    faces = detector.detect(image_path)
    if not faces:
        raise ValueError(f"no faces detected in {image_path}")

    chosen = _pick_face_index(faces, face_index)
    target = faces[chosen]

    refs = ReferenceLibrary(settings.face_references_dir)
    total_refs = refs.append(name, target.embedding[None, :])

    store = EmbeddingStore(settings.face_embeddings_npy, settings.face_embeddings_meta)
    parquet_row_updated = False
    xmp_written = False

    matching_row = _find_matching_parquet_row(
        str(image_path), target.embedding, store
    )
    if matching_row is not None:
        store.update_assignments(
            labels={matching_row: name},
            cluster_ids={matching_row: -2},
        )
        parquet_row_updated = True
        with open_state(settings.state_db) as state:
            written = sync_xmp_for_images(
                {str(image_path)}, store=store, state_conn=state
            )
        xmp_written = written > 0

    return AddImageReport(
        image_path=str(image_path),
        faces_detected=len(faces),
        chosen_face_index=chosen,
        reference_count_after=total_refs,
        parquet_row_updated=parquet_row_updated,
        xmp_written=xmp_written,
    )
