# dt-aid

AI-assisted auto-tagging for [darktable](https://www.darktable.org) libraries. Reads your darktable `library.db`, runs face recognition (InsightFace) and zero-shot element tagging (OpenCLIP) against the images, and writes results back as XMP sidecar tags.

> **Safety:** `dt-aid` never writes to `library.db` by default. It writes XMP sidecars that darktable picks up on its next scan or re-import. Direct DB writes are planned but opt-in and gated behind a running-darktable check.

## Status

- v1 in progress: XMP writes only.
- Two frontends ship: the `dt-aid` CLI and a darktable lighttable Lua plugin (see `plugins/darktable-lua/INSTALL.md`).
- Direct `library.db` writes: not yet implemented.

## Install

Requires Python 3.11+.

```bash
# with uv (recommended)
uv venv
uv pip install -e .

# for element tagging (CLIP)
uv pip install -e '.[elements]'

# for GPU-accelerated face detection
uv pip install -e '.[gpu]'
```

On first face-detection run, InsightFace downloads the `buffalo_l` model pack (~300MB) to `~/.local/share/dt-aid/models/`.

## Quickstart

**1. Close darktable.** The tool refuses to proceed if `darktable` is running, because concurrent access risks corrupting `library.db`.

**2. (Optional) Prepare known faces.**

```
known-faces/
├── alice/
│   ├── 1.jpg
│   └── 2.jpg
└── bob/
    └── 1.jpg
```

```bash
dt-aid faces build-refs ./known-faces
```

**3. Scan.**

```bash
# dry run — see what would happen
dt-aid scan --faces --elements --dry-run

# write XMP sidecars for real
dt-aid scan --faces --elements --write xmp

# scan only a subtree of your library
dt-aid scan --faces --path /photos/2024
```

**4. In darktable:** make darktable pick up the new tags. The simplest way is the one-time preference:

- Open **Preferences → Storage** and enable **"look for updated XMP files on startup"**, then restart darktable. From then on, any sidecar newer than darktable's DB record is re-read automatically on launch.

If you don't want to restart or want a one-off reload: in **lighttable**, select the affected images → right-click → **"selected image(s) → reload the selected XMP files"**. Managed tags then appear under the `people|*` and `auto|*` hierarchies in the tagging panel.

**5. Group the unknowns into clusters.**

A scan leaves unmatched faces "unassigned" — it does not cluster them automatically. Run:

```bash
dt-aid faces cluster
```

This groups visually similar unmatched faces into `people|unknown|cluster-NNN` and rewrites the affected XMPs.

**6. Promote a cluster to a named person.**

```bash
dt-aid faces relabel 7 alice
```

Writes `alice` as a reference and flips every face in cluster 7 to `people|alice`.

**7. Propagate the new reference across the rest of the library.**

```bash
dt-aid faces rematch
```

Re-runs only the matcher (no re-detection, no GPU) against every still-unlabeled face and promotes any that now match. Seconds on a 30k-face library.

See the [user guide](docs/user-guide.md) for the full end-to-end flow, including adding new images, enriching a reference with a single extra photo, and the recommended scan → cluster → relabel → rematch cycle.

## Tag namespace

| Root | Pattern | Example |
|---|---|---|
| `people` | `people\|<name>` | `people\|alice` |
| `people` (unknown) | `people\|unknown\|cluster-<N>` | `people\|unknown\|cluster-007` |
| `auto` (objects) | `auto\|object\|<label>` | `auto\|object\|dog` |
| `auto` (scenes) | `auto\|scene\|<label>` | `auto\|scene\|beach` |
| `auto` (attributes) | `auto\|attr\|<label>` | `auto\|attr\|sunset` |
| `auto` (provenance) | `auto\|_meta\|model-<kind>-<version>` | `auto\|_meta\|model-faces-buffalo-l-v1` |

Tags outside `people|*` and `auto|*` are never modified.

## Data locations

- Darktable library (read-only): `~/.config/darktable/library.db` (Linux) or `~/Library/application support/darktable/library.db` (macOS)
- `dt-aid` state and caches: `~/.local/share/dt-aid/`
  - `state.db` — processed-image ledger and cluster labels
  - `faces/embeddings.npy` + `faces/embeddings.meta.parquet` — face embedding cache
  - `faces/references/<name>.npy` — per-person reference vectors
  - `models/` — downloaded InsightFace + CLIP weights

## Configuration

Override defaults with environment variables (`DT_AID_` prefix) or CLI flags:

| Setting | Env var | Default |
|---|---|---|
| Darktable library path | `DT_AID_DARKTABLE_LIBRARY` | platform default |
| Data dir | `DT_AID_DATA_DIR` | XDG user data / `dt-aid` |
| Face match threshold | `DT_AID_FACE_MATCH_THRESHOLD` | 0.5 |
| Face detection min score | `DT_AID_FACE_DET_SCORE_THRESHOLD` | 0.5 |
| Element tag threshold | `DT_AID_ELEMENTS_THRESHOLD` | 0.25 |
| Elements batch size (CLIP) | `DT_AID_ELEMENTS_BATCH_SIZE` | 16 |
| Inference device | `DT_AID_DEVICE` | auto (CUDA if available else CPU) |

## Development

```bash
uv pip install -e '.[dev]'
pytest
```

The smoke tests build a miniature darktable-shaped fixture library and verify the DB reader, tagging rules, XMP round-trip, and state ledger without requiring the ML model weights.

## Command reference

| Command | Purpose |
|---|---|
| `dt-aid scan --faces [--elements]` | Detect faces (and optionally CLIP element tags) on new images; write XMP sidecars. |
| `dt-aid faces build-refs <dir>` | Build per-person reference embeddings from a directory of labeled photos (overwrites existing references). |
| `dt-aid faces add-image <image> <name>` | Teach dt-aid that a specific face in one image is `<name>`; appends to the reference. |
| `dt-aid faces cluster` | Group the currently unmatched faces into `people\|unknown\|cluster-NNN`. |
| `dt-aid faces relabel <id> <name>` | Promote a cluster to a named person and save its vectors as a reference. |
| `dt-aid faces rematch` | Re-run the matcher over every unlabeled face against the current references (fast; no re-detection). |
| `dt-aid faces list` | JSON dump of references + clusters (for tooling/plugins). |

## Architecture

The core library (`dt_aid.core`) is frontend-agnostic — no CLI imports. Two frontends ship today: the CLI and a darktable lighttable Lua plugin. A third frontend (web UI, GUI, etc.) can either import `dt_aid.core.pipeline.scan()` directly or shell out to the CLI's JSON-emitting commands.

## Documentation

- [User guide](docs/user-guide.md) — workflows and when/how to run each command (start here)
- [Architecture](docs/architecture.md) — components, data flow, safety model
- [Tag namespace](docs/tag-namespace.md) — slug rules, merge semantics, hierarchy expansion
- [Data formats](docs/data-formats.md) — `state.db` schema, embedding store, XMP layout
- [Development](docs/development.md) — setup, tests, invariants, extending
- [Lua plugin install](plugins/darktable-lua/INSTALL.md)
- [Changelog](docs/CHANGELOG.md)

## Limitations

- **v1 writes XMP only.** Direct `library.db` writes are planned.
- **Running darktable is blocked.** Close darktable before scanning.
- **Raw formats** are supported as long as your OS's libraries can decode them via OpenCV/Pillow. Very exotic raws may need a pre-processing step to emit a viewable JPEG/TIFF alongside.
- **Clustering is a manual step.** `dt-aid scan` does not re-cluster unmatched faces; run `dt-aid faces cluster` after a scan (or whenever new unmatched faces have accumulated).
