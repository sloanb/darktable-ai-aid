# Data formats

Everything `dt-aid` persists lives under `~/.local/share/dt-aid/` (XDG user data directory). This directory is **never** inside any darktable directory.

```
~/.local/share/dt-aid/
├── state.db                         # SQLite — processed images + cluster labels
├── faces/
│   ├── embeddings.npy               # float32 [N, 512], L2-normalized
│   ├── embeddings.meta.parquet      # one row per face
│   └── references/
│       └── <name>.npy               # per-person reference vectors
└── models/
    └── models/buffalo_l/*.onnx      # InsightFace weights (downloaded on first use)
```

## `state.db`

Schema in `core.state`:

```sql
CREATE TABLE processed (
    image_path             TEXT PRIMARY KEY,
    dt_image_id            INTEGER,
    faces_model_version    TEXT,     -- e.g. "buffalo_l-v1"
    elements_model_version TEXT,     -- e.g. "openclip-ViT-B-32-...-v1"
    processed_at           TEXT NOT NULL
);

CREATE TABLE clusters (
    cluster_id   INTEGER PRIMARY KEY,
    label        TEXT,                -- NULL until user promotes
    promoted_at  TEXT
);
```

**Idempotency contract:** a row in `processed` means "this image was seen at these model versions." `needs_processing()` returns `(False, False)` if both versions match the current constants, which is how re-scans avoid redundant work. `--force` bypasses the check.

## `faces/embeddings.npy` + `.meta.parquet`

The two files are **row-aligned**: row `k` in the parquet refers to row `k` in the npy array.

`.npy`:
- shape `(N, 512)`, `dtype float32`
- every row is L2-normalized (`‖v‖₂ = 1`) so cosine similarity is just a dot product
- append-only; resaved atomically on each batch via `.npy.tmp` rename

`.meta.parquet` schema (pyarrow):

| Column | Type | Meaning |
|---|---|---|
| `row` | int64 | Index into the npy file (equals the parquet row order) |
| `image_path` | string | Absolute path to the source image |
| `dt_image_id` | int64 | darktable `images.id`, or -1 if unknown |
| `bbox_x1`…`bbox_y2` | float32 | Face bounding box in source-image pixel coordinates |
| `det_score` | float32 | RetinaFace detection confidence |
| `cluster_id` | int64 | `-1` = unassigned / pending cluster pass, `-2` = matched to a known person, `≥0` = HDBSCAN cluster |
| `label` | string | Known-person name if `cluster_id == -2`, else `""` |

**Invariants:**
- Rows for a given image are written together and only removed together via `delete_rows_for_images(paths)`. That method is the single place where rows are pruned; it is used by the `--force` rescan path to avoid double-counting detections.
- Relabeling updates `label` and `cluster_id` in place via `update_assignments()` (pure-pyarrow, no pandas).
- After any `delete_rows_for_images()`, the `row` column is renumbered to match the new npy positions (`0..N-1`). External consumers that persist row IDs (currently none; `state.db` does not) would need to be refreshed.
- The file pair is the authoritative face history. `state.db` is convenience metadata that can be rebuilt from darktable + the parquet.

## `faces/references/<name>.npy`

Per-person reference vectors: shape `(K, 512)`, `dtype float32`, L2-normalized. Matching uses **max** cosine similarity over a person's K references, so adding more varied reference shots improves recall without hurting precision.

Reference files are written by:
- `dt-aid faces build-refs <dir>` — one file per subdirectory, one vector per photo. **Overwrites** an existing `<name>.npy` (via `ReferenceLibrary.save()`).
- `dt-aid faces relabel <cluster-id> <name>` — copies every embedding in that cluster into the new reference. **Overwrites** (via `save()`).
- `dt-aid faces add-image <image> <name>` — detects faces in one image and **appends** the chosen face's embedding to `<name>.npy` (via `ReferenceLibrary.append()`). Use this to enrich an existing person without clobbering prior vectors.

## XMP sidecars

`dt-aid` writes two XMP properties per image, both as `rdf:Bag`:

```xml
<dc:subject>
  <rdf:Bag>
    <rdf:li>parker</rdf:li>
    <rdf:li>dog</rdf:li>
  </rdf:Bag>
</dc:subject>
<lr:hierarchicalSubject>
  <rdf:Bag>
    <rdf:li>people</rdf:li>
    <rdf:li>people|parker</rdf:li>
    <rdf:li>auto</rdf:li>
    <rdf:li>auto|object</rdf:li>
    <rdf:li>auto|object|dog</rdf:li>
  </rdf:Bag>
</lr:hierarchicalSubject>
```

- `dc:subject` holds leaf labels only (compatible with Lightroom/digiKam/etc.).
- `lr:hierarchicalSubject` holds every ancestor of every managed tag, which is how darktable reconstructs the tree view.
- Sidecar path = `<image>.<ext>.xmp` (darktable convention; the sidecar sits next to the image).
- `core.xmp.write_subjects()` preserves unknown XMP properties in existing sidecars (we only rewrite `dc:subject` and `lr:hierarchicalSubject`).

## Darktable library reads

`dt-aid` reads but never writes:

- `main.images` — `id`, `film_id`, `filename`
- `main.film_rolls` — `id`, `folder`
- `main.tagged_images` — `imgid`, `tagid`
- `main.tags` **or** `data.tags` — `id`, `name` (darktable 3.0+ moved this to a sibling `data.db`; we attach it read-only and sniff whichever schema has the table)

See `core.darktable_db._tags_schema()` for the sniff logic.
