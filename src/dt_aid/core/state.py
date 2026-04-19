from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS processed (
    image_path TEXT PRIMARY KEY,
    dt_image_id INTEGER,
    faces_model_version TEXT,
    elements_model_version TEXT,
    processed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER PRIMARY KEY,
    label TEXT,
    promoted_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_clusters_label ON clusters(label);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def open_state(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_processed(
    conn: sqlite3.Connection, image_path: str
) -> sqlite3.Row | None:
    cur = conn.execute(
        "SELECT * FROM processed WHERE image_path = ?", (image_path,)
    )
    return cur.fetchone()


def needs_processing(
    conn: sqlite3.Connection,
    image_path: str,
    *,
    faces_version: str | None,
    elements_version: str | None,
    force: bool = False,
) -> tuple[bool, bool]:
    """Return (run_faces, run_elements) booleans."""
    if force:
        return (faces_version is not None, elements_version is not None)
    row = get_processed(conn, image_path)
    if row is None:
        return (faces_version is not None, elements_version is not None)
    run_faces = faces_version is not None and row["faces_model_version"] != faces_version
    run_elements = (
        elements_version is not None
        and row["elements_model_version"] != elements_version
    )
    return (run_faces, run_elements)


def mark_processed(
    conn: sqlite3.Connection,
    image_path: str,
    *,
    dt_image_id: int | None,
    faces_version: str | None,
    elements_version: str | None,
) -> None:
    existing = get_processed(conn, image_path)
    new_faces = faces_version if faces_version is not None else (
        existing["faces_model_version"] if existing else None
    )
    new_elements = elements_version if elements_version is not None else (
        existing["elements_model_version"] if existing else None
    )
    conn.execute(
        """
        INSERT INTO processed (image_path, dt_image_id, faces_model_version,
                               elements_model_version, processed_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(image_path) DO UPDATE SET
            dt_image_id = excluded.dt_image_id,
            faces_model_version = excluded.faces_model_version,
            elements_model_version = excluded.elements_model_version,
            processed_at = excluded.processed_at
        """,
        (image_path, dt_image_id, new_faces, new_elements, _now()),
    )


def upsert_cluster(conn: sqlite3.Connection, cluster_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO clusters (cluster_id, label, promoted_at) VALUES (?, NULL, NULL)",
        (cluster_id,),
    )


def promote_cluster(conn: sqlite3.Connection, cluster_id: int, label: str) -> None:
    conn.execute(
        """
        INSERT INTO clusters (cluster_id, label, promoted_at)
        VALUES (?, ?, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET label = excluded.label,
                                              promoted_at = excluded.promoted_at
        """,
        (cluster_id, label, _now()),
    )


def cluster_label(conn: sqlite3.Connection, cluster_id: int) -> str | None:
    row = conn.execute(
        "SELECT label FROM clusters WHERE cluster_id = ?", (cluster_id,)
    ).fetchone()
    return row["label"] if row else None
