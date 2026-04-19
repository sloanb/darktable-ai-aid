from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

EMBED_DIM = 512

META_SCHEMA = pa.schema(
    [
        ("row", pa.int64()),
        ("image_path", pa.string()),
        ("dt_image_id", pa.int64()),
        ("bbox_x1", pa.float32()),
        ("bbox_y1", pa.float32()),
        ("bbox_x2", pa.float32()),
        ("bbox_y2", pa.float32()),
        ("det_score", pa.float32()),
        ("cluster_id", pa.int64()),  # -1 = unassigned, -2 = matched known person
        ("label", pa.string()),       # person name if matched, else ""
    ]
)


@dataclass
class FaceRow:
    image_path: str
    dt_image_id: int | None
    bbox: tuple[float, float, float, float]
    det_score: float
    embedding: np.ndarray
    cluster_id: int = -1
    label: str = ""


class EmbeddingStore:
    """
    Append-only face embedding store:
      - embeddings.npy     : float32 [N, 512], L2-normalized (plain .npy, re-saved on append)
      - embeddings.meta.parquet : one row per face, aligned by `row` index
    """

    def __init__(self, npy_path: Path, meta_path: Path) -> None:
        self.npy_path = npy_path
        self.meta_path = meta_path

    def load_embeddings(self) -> np.ndarray:
        if not self.npy_path.exists():
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        return np.load(self.npy_path, mmap_mode="r")

    def load_meta(self) -> pa.Table:
        if not self.meta_path.exists():
            return pa.Table.from_pylist([], schema=META_SCHEMA)
        return pq.read_table(self.meta_path)

    def append(self, rows: list[FaceRow]) -> None:
        if not rows:
            return
        self.npy_path.parent.mkdir(parents=True, exist_ok=True)

        existing = (
            np.load(self.npy_path) if self.npy_path.exists()
            else np.zeros((0, EMBED_DIM), dtype=np.float32)
        )
        new_vecs = np.stack([r.embedding.astype(np.float32) for r in rows])
        combined = np.concatenate([existing, new_vecs], axis=0)
        # write atomically. Pass a file handle so np.save doesn't silently
        # append its own ".npy" suffix (which it does for path-like args).
        tmp = self.npy_path.parent / (self.npy_path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, combined)
        tmp.replace(self.npy_path)

        base = existing.shape[0]
        records = [
            {
                "row": base + i,
                "image_path": r.image_path,
                "dt_image_id": r.dt_image_id if r.dt_image_id is not None else -1,
                "bbox_x1": float(r.bbox[0]),
                "bbox_y1": float(r.bbox[1]),
                "bbox_x2": float(r.bbox[2]),
                "bbox_y2": float(r.bbox[3]),
                "det_score": float(r.det_score),
                "cluster_id": int(r.cluster_id),
                "label": r.label,
            }
            for i, r in enumerate(rows)
        ]
        new_table = pa.Table.from_pylist(records, schema=META_SCHEMA)
        if self.meta_path.exists():
            existing_meta = pq.read_table(self.meta_path)
            merged = pa.concat_tables([existing_meta, new_table])
        else:
            merged = new_table
        tmp_meta = self.meta_path.with_suffix(".parquet.tmp")
        pq.write_table(merged, tmp_meta)
        tmp_meta.replace(self.meta_path)

    def update_assignments(
        self,
        *,
        cluster_ids: dict[int, int] | None = None,
        labels: dict[int, str] | None = None,
    ) -> None:
        """Update cluster_id / label columns by row index. No-op if empty."""
        if not (cluster_ids or labels):
            return
        if not self.meta_path.exists():
            return
        table = pq.read_table(self.meta_path)
        rows = table.column("row").to_pylist()
        cluster_col = table.column("cluster_id").to_pylist()
        label_col = table.column("label").to_pylist()
        if cluster_ids:
            for i, r in enumerate(rows):
                if r in cluster_ids:
                    cluster_col[i] = int(cluster_ids[r])
        if labels:
            for i, r in enumerate(rows):
                if r in labels:
                    label_col[i] = labels[r]
        new_table = table.set_column(
            table.schema.get_field_index("cluster_id"),
            "cluster_id",
            pa.array(cluster_col, type=pa.int64()),
        ).set_column(
            table.schema.get_field_index("label"),
            "label",
            pa.array(label_col, type=pa.string()),
        )
        tmp = self.meta_path.with_suffix(".parquet.tmp")
        pq.write_table(new_table, tmp)
        tmp.replace(self.meta_path)

    def delete_rows_for_images(self, image_paths: set[str]) -> int:
        """
        Atomically remove all face rows belonging to the given image paths
        and renumber surviving rows so the `row` column stays aligned with
        the npy array. Returns the number of rows removed.

        Intended for `--force` rescans: deleting the stale rows up-front
        prevents the append-only store from double-counting detections.
        """
        if not image_paths:
            return 0
        if not self.meta_path.exists() or not self.npy_path.exists():
            return 0
        meta = pq.read_table(self.meta_path)
        paths_array = pa.array(list(image_paths), type=pa.string())
        keep_mask = pc.invert(pc.is_in(meta["image_path"], value_set=paths_array))
        keep = meta.filter(keep_mask)
        removed = meta.num_rows - keep.num_rows
        if removed == 0:
            return 0

        all_vecs = np.load(self.npy_path)
        keep_row_ids = keep["row"].to_numpy()
        new_vecs = all_vecs[keep_row_ids] if keep.num_rows else np.zeros((0, EMBED_DIM), dtype=np.float32)

        # Renumber row column to match new npy positions.
        new_meta = keep.set_column(
            keep.schema.get_field_index("row"),
            "row",
            pa.array(list(range(keep.num_rows)), type=pa.int64()),
        )

        tmp_npy = self.npy_path.parent / (self.npy_path.name + ".tmp")
        with open(tmp_npy, "wb") as f:
            np.save(f, new_vecs)
        tmp_npy.replace(self.npy_path)

        tmp_meta = self.meta_path.with_suffix(".parquet.tmp")
        pq.write_table(new_meta, tmp_meta)
        tmp_meta.replace(self.meta_path)
        return removed


class ReferenceLibrary:
    """
    Per-person reference embeddings. Each known person gets one file
    <name>.npy containing a [K, 512] array of reference vectors. Matching
    uses max cosine similarity across that person's vectors.
    """

    def __init__(self, references_dir: Path) -> None:
        self.dir = references_dir

    def names(self) -> list[str]:
        if not self.dir.exists():
            return []
        return sorted(p.stem for p in self.dir.glob("*.npy"))

    def load(self, name: str) -> np.ndarray:
        return np.load(self.dir / f"{name}.npy")

    def save(self, name: str, vectors: np.ndarray) -> None:
        """Replace any existing reference for `name` with these vectors."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = (vectors / norms).astype(np.float32)
        path = self.dir / f"{name}.npy"
        tmp = self.dir / f"{name}.npy.tmp"
        with open(tmp, "wb") as f:
            np.save(f, normalized)
        tmp.replace(path)

    def append(self, name: str, vectors: np.ndarray) -> int:
        """
        Merge vectors into the existing reference for `name`, or create it
        if missing. Returns the total vector count after the append. Does
        not deduplicate — callers that want unique references should pass
        distinct embeddings.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new = (vectors / norms).astype(np.float32)
        path = self.dir / f"{name}.npy"
        if path.exists():
            existing = np.load(path)
            combined = np.concatenate([existing, new], axis=0)
        else:
            combined = new
        tmp = self.dir / f"{name}.npy.tmp"
        with open(tmp, "wb") as f:
            np.save(f, combined)
        tmp.replace(path)
        return int(combined.shape[0])

    def load_all(self) -> dict[str, np.ndarray]:
        return {n: self.load(n) for n in self.names()}
