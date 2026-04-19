"""
Build a tiny darktable-shaped SQLite library plus matching JPEG files.

This is NOT the full darktable schema — it only populates the tables and
columns that `dt_aid.core.darktable_db` reads from.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from PIL import Image

SCHEMA = """
CREATE TABLE film_rolls (
    id INTEGER PRIMARY KEY,
    folder TEXT NOT NULL
);
CREATE TABLE images (
    id INTEGER PRIMARY KEY,
    film_id INTEGER NOT NULL,
    filename TEXT NOT NULL
);
CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE tagged_images (
    imgid INTEGER NOT NULL,
    tagid INTEGER NOT NULL,
    PRIMARY KEY (imgid, tagid)
);
"""


def _make_jpeg(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")


def build(root: Path) -> tuple[Path, list[Path]]:
    """
    Create `<root>/library.db` and a film roll at `<root>/photos/` with
    three JPEGs. Returns (library_db_path, image_paths).
    """
    root.mkdir(parents=True, exist_ok=True)
    photos = root / "photos"
    photos.mkdir(parents=True, exist_ok=True)

    images = [
        (photos / "red.jpg", (200, 30, 30)),
        (photos / "green.jpg", (30, 200, 30)),
        (photos / "blue.jpg", (30, 30, 200)),
    ]
    for p, color in images:
        _make_jpeg(p, color)

    db_path = root / "library.db"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO film_rolls (id, folder) VALUES (?, ?)", (1, str(photos)))
        for i, (p, _) in enumerate(images, start=1):
            conn.execute(
                "INSERT INTO images (id, film_id, filename) VALUES (?, ?, ?)",
                (i, 1, p.name),
            )
        # seed one user tag to verify we don't clobber non-managed tags
        conn.execute("INSERT INTO tags (id, name) VALUES (?, ?)", (1, "keepers"))
        conn.execute("INSERT INTO tagged_images (imgid, tagid) VALUES (?, ?)", (1, 1))
        conn.commit()
    finally:
        conn.close()

    return db_path, [p for p, _ in images]


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scratch/fixture_library")
    db, imgs = build(target)
    print(f"library: {db}")
    for p in imgs:
        print(f"  {p}")
