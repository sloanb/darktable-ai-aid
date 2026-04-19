# Changelog

All notable changes to `dt-aid`. The format loosely follows [Keep a Changelog](https://keepachangelog.com/) and dates are ISO-8601.

## Unreleased

### Fixed — 2026-04-19
- **Scanning one detector no longer wipes the other's tags.** Running `dt-aid scan --elements` on a library previously tagged by `scan --faces` (or vice versa) silently stripped every `people|*` tag from affected XMPs, because the pipeline's write path called `merge_managed()` with a `new_tags` list that only contained the re-running detector's output. The skipped detector's tags are now carried over from the existing XMP before `merge_managed()` runs, gated by `tagging.is_face_tag()` / `tagging.is_elements_tag()`. Regression test: `test_pipeline_merge_logic_preserves_uninvolved_detector_tags` (and its symmetric counterpart). Users who were avoiding incremental elements scans to protect people tags can now run them safely without `--force`.

### Added
- 2026-04-18 — Initial scaffold: `core` + `cli` split, pyproject with `uv`-friendly extras (`elements`, `gpu`, `yolo`, `dev`).
- 2026-04-18 — `core.darktable_db`: read-only library reader with running-darktable guard.
- 2026-04-18 — `core.xmp`: sidecar read/write for `dc:subject` + `lr:hierarchicalSubject`, hierarchy expansion.
- 2026-04-18 — `core.tagging`: two-root namespace (`people|*`, `auto|*`), slug helper, managed-tag merge.
- 2026-04-18 — `core.faces`: InsightFace `buffalo_l` detector, memmapped `.npy` + Parquet embedding store, cosine matcher, HDBSCAN clustering.
- 2026-04-18 — `core.elements.clip_tagger`: OpenCLIP zero-shot tagger (behind `.[elements]` extra).
- 2026-04-18 — `core.state`: SQLite ledger with per-model-version idempotency.
- 2026-04-18 — CLI: `dt-aid scan`, `dt-aid faces build-refs`, `dt-aid faces relabel`, JSON logs, rich progress.
- 2026-04-18 — Smoke test suite: synthetic darktable fixture, 6 tests covering DB, tagging, XMP, state.
- 2026-04-18 — `core.image_io`: **RAW support via `rawpy`.** Extracts embedded JPEG previews (10–50× faster than demosaicing). Handles DNG, RAF, CR2/CR3, NEF, ARW, ORF, RW2, PEF, SRW, X3F, and more.
- 2026-04-18 — Developer documentation: `docs/{architecture,tag-namespace,data-formats,development}.md`.

### Fixed
- 2026-04-18 — **darktable 3.0+ schema split.** `tags` moved from `library.db` to a sibling `data.db`; reader now attaches `data.db` read-only and sniffs whichever schema owns the `tags` table.
- 2026-04-18 — CLI dry-run preview: previously showed the first 10 results regardless of content, so scans where the earliest images had no detections appeared empty. Now samples from the subset with new tags and reports the count.
- 2026-04-18 — **Embedding store atomic write.** `np.save(path, arr)` silently appends `.npy` to path-like arguments, so the "atomic write" was creating `embeddings.npy.tmp.npy` and the subsequent rename failed with ENOENT on the first real (non-dry-run) scan. Fixed by passing an open file handle to `np.save`, which preserves the exact tmp filename.
- 2026-04-18 — `dt-aid faces build-refs` crashed with `TypeError: got multiple values for keyword argument 'known_faces_dir'` because the CLI was passing the directory both via `_settings_overrides()` (which reads `args.known_faces`) and as an explicit kwarg. Removed the duplicate.
- 2026-04-18 — **`--force` no longer doubles the embedding cache.** The store is append-only, so a `--force` rescan used to leave stale rows from the previous pass — a 32k-face library would grow to 65k after one rerun, slowing each `append()` (whole-file rewrite). `pipeline.scan()` now calls `EmbeddingStore.delete_rows_for_images()` once up-front for every image that exists on disk before the detection loop runs, then appends fresh rows. The `row` column is renumbered so it stays aligned with the npy positions.
- 2026-04-18 — `EmbeddingStore.update_assignments()` no longer requires `pandas`. Rewritten with pure pyarrow so `dt-aid faces relabel` works in the default install.
- 2026-04-18 — `dt-aid faces relabel` also no longer requires `pandas`; the CLI handler was calling `.to_pandas()` directly in addition to the store method. Rewritten with pure pyarrow.

### Changed (continued)
- 2026-04-18 — **`dt-aid faces relabel` now re-syncs XMP sidecars.** Previously it only updated the parquet + state.db, so images kept their `people|unknown|cluster-NNN` tag in darktable until the next full scan. It now calls `xmp_sync.sync_xmp_for_images()` for every image touched by the cluster, replacing the cluster tag with `people|<name>`. The relabel operation also flips `cluster_id` from the cluster number to -2 (matched-known) so the promoted rows are semantically identical to freshly-matched faces.

### Added (continued 3)
- 2026-04-18 — **`dt-aid faces rematch` command.** Runs the matcher over every unlabeled face embedding (noise + unpromoted clusters) against the current `references/` library. Rows that exceed the threshold get `label = <name>, cluster_id = -2` in the parquet and their XMPs are re-synced. No GPU, no re-detection — the intended partner to `build-refs` and `relabel`. Flags: `--threshold N` (defaults to `DT_AID_FACE_MATCH_THRESHOLD`), `--no-write-xmp`.

### Added (continued 4) — 2026-04-19
- **`dt-aid faces add-image <image> <name>` command.** Teaches dt-aid that a specific face in a specific image is a named person. Detects faces, picks the largest (or `--face-index N`), **appends** its embedding to `references/<name>.npy` (merge, not overwrite), and if the image is already in the embedding store updates its row to `label=<name>, cluster_id=-2` and re-syncs the XMP. `--json` flag emits a one-line machine-readable summary for tooling.
- **`ReferenceLibrary.append(name, vectors)`.** Merge-into-existing counterpart to `save()`. Atomic write via `.npy.tmp` rename. Foundation for `add-image`; partially closes the "save() overwrites" gotcha flagged in earlier deferred work.
- **`dt-aid faces list` command (JSON).** Emits `{ "references": [...], "clusters": [...] }` for tooling. References include vector counts; clusters include face count, distinct image count, and any promoted label. `--min-size N` filters small clusters.
- **darktable Lua plugin (`plugins/darktable-lua/dt-aid.lua`).** Lighttable module with a full phase 1 + phase 2 surface:
  - Buttons: *scan*, *cluster*, *rematch*, *refresh names + clusters*.
  - Name chooser: combobox of existing references (populated via `faces list`) + free-text entry for new names.
  - Label-selected actions: *tag selected as…* (calls `faces add-image` per image, then runs `rematch`), *promote cluster of selected* (reads the `people|unknown|cluster-NNN` tag from the selected image and calls `faces relabel`).
  - Cluster review: dropdown of clusters sorted by size, *filter grid to cluster* button, *promote selected cluster* button.
  - Preferences: *dt-aid: binary path* (string) and *dt-aid: inference device* (auto/cpu/cuda).
  - Inline minimal JSON decoder (~50 lines) so no external Lua library is required.
  See `plugins/darktable-lua/INSTALL.md`.

### Added (continued 2)
- 2026-04-18 — `EmbeddingStore.delete_rows_for_images(paths)`: atomic batch delete + renumber of the embedding cache. Paired with the `--force` fix above; also usable by future "reset" or "rematch" commands.
- 2026-04-18 — **`dt-aid faces cluster` command.** Runs HDBSCAN over every face with `cluster_id = -1`, assigns `cluster_id ≥ 0` to each dense group, writes IDs back to the parquet and the `clusters` table in `state.db`, then re-syncs XMP sidecars so affected images gain a `people|unknown|cluster-NNN` tag. Flags: `--min-cluster-size N` (default 5), `--min-samples N`, `--no-write-xmp`. No GPU required — pure embedding-space math on cached vectors.
- 2026-04-18 — `core.xmp_sync.sync_xmp_for_images(paths, ...)`: reusable helper that derives the full managed tag set for an image from its parquet rows (matched labels, cluster IDs, promoted cluster labels) and rewrites the XMP sidecar, preserving non-managed tags. Used by `faces cluster`; also the natural home for a future `faces rematch` command.

### Changed
- 2026-04-18 — `cluster_unknowns()` rewritten to use HDBSCAN with the euclidean metric on L2-normalized vectors (O(N log N) memory via ball-tree) instead of a precomputed cosine-distance matrix (which would have been ~7 GB RAM for a 32 k-embedding library). Mathematically equivalent for unit vectors via `‖a − b‖² = 2 − 2·a·b`. Default `min_cluster_size` raised from 3 → 5 to avoid micro-clusters in large libraries.

### Docs
- 2026-04-18 — README quickstart now explains how to make darktable re-read XMP sidecars after a scan (preference toggle **Storage → "look for updated XMP files on startup"**, or one-off per-image reload). The tool writes the sidecars; darktable picks them up on the next scan, not automatically.

### Added (continued)
- 2026-04-18 — **GPU support** via `core.device`. Face detection (InsightFace / ONNX Runtime) and element tagging (CLIP / torch) now honor a `device` setting (`auto` | `cpu` | `cuda`). New CLI flag `--device` on `scan` and `faces build-refs`. `auto` picks CUDA when `onnxruntime-gpu` is importable and a CUDA provider is registered, else falls back to CPU with a log line. Expected 10–30× face-detection speedup on NVIDIA hardware; CLIP throughput also benefits.
- 2026-04-18 — **Auto-preload of bundled CUDA 12 runtime libs.** `core.device.preload_cuda_libs()` dlopens the `nvidia-*-cu12` shared libraries from the venv's site-packages before ONNX Runtime initializes its CUDA provider, so `dt-aid scan --device cuda` works without the user setting `LD_LIBRARY_PATH`. The `[gpu]` extra now pins the nvidia CUDA 12 packages explicitly; Blackwell (SM_120) requires `onnxruntime-gpu>=1.24`.
