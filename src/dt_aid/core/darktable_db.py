from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import psutil


class DarktableRunningError(RuntimeError):
    """Raised when a darktable process is detected and DB access is unsafe."""


@dataclass
class DtImage:
    id: int
    path: Path
    filename: str
    film_id: int
    tags: list[str] = field(default_factory=list)


def _tags_schema(conn: sqlite3.Connection) -> str | None:
    """
    Return the schema name (main or data) that owns a `tags` table, or
    None if neither exists. Modern darktable (3.0+) stores tags in
    data.db; older databases kept tags inside library.db.
    """
    for schema in ("main", "data"):
        try:
            row = conn.execute(
                f"SELECT name FROM {schema}.sqlite_master "
                "WHERE type='table' AND name='tags'"
            ).fetchone()
        except sqlite3.OperationalError:
            continue
        if row is not None:
            return schema
    return None


def is_darktable_running() -> bool:
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name in {"darktable", "darktable-cli"}:
                return True
            cmdline = proc.info.get("cmdline") or []
            if cmdline and Path(cmdline[0]).name.lower() in {"darktable", "darktable-cli"}:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """
    Open darktable's library.db read-only. If a sibling data.db exists
    (modern darktable layout), attach it as schema `data` so tag names
    can be resolved from `data.tags`.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"darktable library not found: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    data_db = db_path.parent / "data.db"
    if data_db.exists():
        conn.execute(f"ATTACH DATABASE 'file:{data_db}?mode=ro' AS data")
    return conn


def iter_images(
    conn: sqlite3.Connection, *, path_prefix: Path | None = None
) -> Iterator[DtImage]:
    """
    Yield all images with resolved on-disk paths and their current tags.

    Darktable schema (current):
      film_rolls(id, folder)
      images(id, film_id, filename)
      tags(id, name)              -- hierarchical with '|' separator
      tagged_images(imgid, tagid)
    """
    query = """
    SELECT i.id AS image_id,
           i.film_id AS film_id,
           i.filename AS filename,
           f.folder AS folder
    FROM images i
    JOIN film_rolls f ON f.id = i.film_id
    ORDER BY i.id
    """
    tags_schema = _tags_schema(conn)
    for row in conn.execute(query):
        path = Path(row["folder"]) / row["filename"]
        if path_prefix is not None:
            try:
                path.relative_to(path_prefix)
            except ValueError:
                continue
        tags = _tags_for_image(conn, row["image_id"], tags_schema)
        yield DtImage(
            id=row["image_id"],
            path=path,
            filename=row["filename"],
            film_id=row["film_id"],
            tags=tags,
        )


def _tags_for_image(
    conn: sqlite3.Connection, image_id: int, tags_schema: str | None
) -> list[str]:
    if tags_schema is None:
        return []
    rows = conn.execute(
        f"""
        SELECT t.name
        FROM tagged_images ti
        JOIN {tags_schema}.tags t ON t.id = ti.tagid
        WHERE ti.imgid = ?
        """,
        (image_id,),
    ).fetchall()
    return [r["name"] for r in rows]


def count_images(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()
    return row["n"] if row else 0
