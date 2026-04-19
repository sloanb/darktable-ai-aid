# Architecture

`dt-aid` is a Python tool that augments a darktable library with AI-derived tags (faces, visual elements). The runtime is split into a **frontend-agnostic core** and two shipping frontends: a CLI (`dt-aid`) and a darktable Lua plugin (`plugins/darktable-lua/dt-aid.lua`).

## Component map

```
                                   ┌───────────────────────┐
                                   │  CLI (dt_aid.cli)     │
                                   │  argparse + rich      │
                                   │  JSON logs, progress  │
                                   └──────────┬────────────┘
                                              │ calls pipeline.scan()
                                              ▼
┌───────────────────────────────────────────────────────────────────────┐
│                       core.pipeline.scan()                            │
│                                                                       │
│  1. Guard: is_darktable_running()  → refuse if yes                    │
│  2. Read: iter_images(library.db)  ─────┐                             │
│  3. For each image:                     │                             │
│       load → detect → match/cluster ─┐  │                             │
│       write XMP sidecar              │  │                             │
│  4. Persist embeddings + state       │  │                             │
└──────┬─────────────────────┬─────────┴──┴─────────────────────────────┘
       │                     │          │
       ▼                     ▼          ▼
┌──────────────┐   ┌──────────────────┐  ┌──────────────────┐
│ image_io     │   │ faces/           │  │ elements/        │
│  cv2 + rawpy │   │  detector        │  │  clip_tagger     │
│  RAW preview │   │  embeddings      │  │  (optional)      │
│  extraction  │   │  matcher         │  │                  │
└──────────────┘   │  cluster         │  └──────────────────┘
                   └──────────────────┘
                            │
                            ▼
                   ┌──────────────────┐  ┌──────────────────┐
                   │ xmp / xmp_sync   │  │ state (sqlite)   │
                   │  dc:subject +    │  │  processed +     │
                   │  lr:hier...      │  │  clusters        │
                   └──────────────────┘  └──────────────────┘
```

## End-to-end data flow

1. **Discovery.** `core.darktable_db.iter_images()` opens `library.db` read-only (`?mode=ro`) and, if present, attaches a sibling `data.db` as schema `data` (darktable 3.0+ split). It joins `images` + `film_rolls` to resolve absolute paths and `tagged_images` + `tags` for pre-existing tags.
2. **Running-darktable guard.** `is_darktable_running()` scans `psutil` for processes named `darktable` / `darktable-cli`. The pipeline raises `DarktableRunningError` if any match, because concurrent writers risk corrupting `library.db`.
3. **Image load.** `core.image_io.load_bgr()` handles JPEG / TIFF / PNG via OpenCV and RAW (DNG, RAF, CR2/CR3, NEF, ARW, ORF, RW2, …) via `rawpy`. For RAW it extracts the embedded JPEG preview (10–50× faster than demosaicing); falls back to half-size LibRaw demosaic when no preview exists.
4. **Face detection + embedding.** `core.faces.detector.FaceDetector` wraps InsightFace `buffalo_l`: RetinaFace produces bounding boxes, ArcFace produces 512-d L2-normalized embeddings. Execution provider (CPU vs CUDA) comes from `core.device.resolve_onnx_providers(settings.device)`; `device="auto"` picks CUDA when `onnxruntime-gpu` is importable, else CPU. CLIP uses the same preference via `resolve_torch_device()`.
5. **Matching.** `core.faces.matcher.FaceMatcher` computes max cosine similarity between each query embedding and the per-person reference matrix built from `~/.local/share/dt-aid/faces/references/<name>.npy`. Above `face_match_threshold` (default 0.5) → `people|<name>` tag.
6. **Clustering (unmatched).** Unmatched face embeddings are stored with `cluster_id = -1`. `dt-aid faces cluster` runs HDBSCAN (euclidean metric on L2-normalized vectors — mathematically equivalent to cosine on unit vectors but uses a ball-tree, so memory is O(N log N) instead of the O(N²) a precomputed distance matrix would need). Rows that land in a dense cluster get `cluster_id ≥ 0`; noise stays at -1. Cluster IDs are persisted in the `clusters` table and rendered on affected XMPs as `people|unknown|cluster-NNN` by `xmp_sync.sync_xmp_for_images()`.

6a. **Rematching after reference changes.** `dt-aid faces rematch` is the fast partner to a full scan: it re-runs the matcher against every unlabeled face embedding (`label == ""`) using the current `references/` library, promotes anything above threshold to `cluster_id = -2, label = <name>`, and re-syncs affected XMPs. No GPU, no re-detection — a ~31 k-face library rematches in seconds. The intended workflow is: `build-refs` (or `relabel` a cluster) → `rematch` → done.
7. **Element tagging (optional).** `core.elements.clip_tagger.ClipTagger` encodes each image with OpenCLIP `ViT-B-32` and scores it against three label groups (objects, scenes, attrs) via softmax-normalized cosine similarity. Labels above `elements_threshold` yield `auto|object|*`, `auto|scene|*`, `auto|attr|*` tags.
8. **Provenance.** Every managed write adds a provenance tag (`auto|_meta|model-faces-<version>` etc.) so re-tagging on model upgrades is detectable.
9. **Write-back — XMP only (v1).** `core.xmp.write_subjects()` writes `dc:subject` (flat leaf labels) and `lr:hierarchicalSubject` (full pipe-joined paths with ancestors expanded) to `<image>.xmp`. `library.db` is never mutated.
10. **Persistence.** `core.state` tracks per-image processing state (`faces_model_version`, `elements_model_version`, `dt_image_id`, `processed_at`) so re-runs skip already-seen images unless `--force` is set or a model version changed. `core.faces.embeddings.EmbeddingStore` appends every detected face to a memmapped `.npy` + Parquet metadata pair, enabling cheap re-clustering and per-person relabeling.

## Safety model

| Concern | Mitigation |
|---|---|
| Corrupting `library.db` | Read-only `file:...?mode=ro` URI. Running-darktable guard. Writes go to XMP sidecars. |
| Clobbering user tags | `tagging.is_managed_tag()` whitelists `people\|*` and `auto\|*`; everything else is preserved verbatim by `merge_managed()`. |
| Model upgrade drift | Provenance tags + per-image model-version tracking in `state.db`. |
| Unreadable RAW formats | `image_io` fallback chain: embedded JPEG → LibRaw demosaic → skip + log. |

## Frontend-agnostic core

`core/` imports nothing from `cli/`. The pipeline accepts a `progress: Callable[[str, int, int], None]` callback so any frontend (CLI, Lua, GUI, tests) can surface progress without coupling to rich/tqdm. Only `cli/` imports rich.

## Frontends

Two frontends ship today:

1. **`dt-aid` CLI** (`dt_aid.cli`) — argparse + rich. Every command accepts `--json` or has a JSON-emitting sibling for tool consumption.
2. **darktable Lua plugin** (`plugins/darktable-lua/dt-aid.lua`) — a lighttable module that shells out to the CLI. Reads `dt-aid faces list` JSON to populate name + cluster dropdowns; invokes `scan`, `faces cluster`, `faces rematch`, `faces add-image`, and `faces relabel` behind buttons. Phase 1 handles scan/match/label workflows; phase 2 adds a cluster review dropdown with grid filtering and one-click promotion. See `plugins/darktable-lua/INSTALL.md`.

The frontend-agnostic core is what makes two (or more) frontends cheap. A third — a web UI, a Qt GUI, a JetBrains plugin — would follow the same pattern: shell out to the CLI's JSON-emitting commands, or import `dt_aid.core` directly.
