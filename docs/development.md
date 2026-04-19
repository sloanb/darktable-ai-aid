# Development

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Optional extras:
- `.[elements]` — OpenCLIP + torch (~2 GB) for zero-shot element tagging. If you run `dt-aid scan --elements` without this extra, the CLI exits with code 5 and prints the install command (see `MissingElementsExtraError` in `core/elements/clip_tagger.py`).
- `.[gpu]` — `onnxruntime-gpu` for CUDA-accelerated face detection
- `.[yolo]` — `ultralytics` for YOLO-based object detection (not yet wired into the pipeline)

### GPU setup

With an NVIDIA GPU (Pascal or newer) you can get 10–30× face-detection speedup:

```bash
# swap onnxruntime for the GPU build
pip uninstall -y onnxruntime
pip install -e '.[gpu]'

# verify CUDA is visible to ORT
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# expect: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

- onnxruntime-gpu PyPI wheels target CUDA 12 but work with newer drivers (CUDA 13.x runtime) via NVIDIA's forward-compatible driver layer. The `[gpu]` extra also pins the `nvidia-*-cu12` pip packages that carry the CUDA and cuDNN shared libraries; these are auto-preloaded by `core.device.preload_cuda_libs()` before ONNX Runtime initializes, so **no `LD_LIBRARY_PATH` setup is required**. System cuDNN is not used.
- **Do not have both `onnxruntime` and `onnxruntime-gpu` installed at the same time** — they share an import name and collide. If insightface or another dep pulls CPU `onnxruntime` back in, uninstall both and reinstall only the GPU variant.
- Blackwell (RTX 50-series, SM_120) needs an onnxruntime-gpu build compiled with SM_120 support. `onnxruntime-gpu>=1.24` includes it; older wheels silently fall back to CPU.
- Select device explicitly with `dt-aid scan --device cuda` or via `DT_AID_DEVICE=cuda`. Default `auto` picks CUDA when the provider is available, else CPU.
- For CLIP (elements tagging), install torch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu128`.

## Tests

```bash
pytest              # runs the smoke suite (no ML weights required)
pytest -v
pytest -k xmp       # run a subset
```

The smoke suite builds a miniature darktable-shaped fixture library (`tests/fixtures/make_library.py`), exercises the DB reader, tagging rules, XMP round-trip, and state ledger. It does **not** require InsightFace, OpenCLIP, or any model weights.

Tests that exercise the real detectors are out of scope for CI because they need model downloads and a real image set. Run those manually against your own library.

## Directory layout

```
src/dt_aid/
├── core/                      # frontend-agnostic
│   ├── config.py              # pydantic settings, XDG paths, model versions
│   ├── state.py               # sqlite ledger for processed/clusters
│   ├── darktable_db.py        # read-only library + data.db access
│   ├── image_io.py            # JPEG/TIFF/PNG via cv2, RAW via rawpy
│   ├── device.py              # device=auto|cpu|cuda + CUDA lib preload
│   ├── tagging.py             # tag namespace, slug, merge_managed
│   ├── xmp.py                 # dc:subject + lr:hierarchicalSubject I/O
│   ├── xmp_sync.py            # regenerate XMP from parquet state
│   ├── pipeline.py            # scan orchestration
│   ├── faces/
│   │   ├── detector.py        # InsightFace buffalo_l wrapper
│   │   ├── embeddings.py      # .npy + parquet store, ReferenceLibrary
│   │   ├── matcher.py         # cosine match
│   │   ├── cluster.py         # HDBSCAN on unit vectors (euclidean)
│   │   ├── cluster_runner.py  # orchestrates faces cluster subcommand
│   │   ├── rematch_runner.py  # orchestrates faces rematch subcommand
│   │   └── add_image_runner.py # orchestrates faces add-image subcommand
│   └── elements/
│       ├── clip_tagger.py     # OpenCLIP zero-shot (optional)
│       └── labels.py          # default label lists + TOML loader
└── cli/
    ├── app.py                 # argparse entrypoint
    ├── logging_setup.py       # JSON logs
    └── progress.py            # rich progress bars

tests/
├── fixtures/make_library.py   # synthetic darktable DB + JPEGs
└── test_smoke.py

plugins/
└── darktable-lua/
    ├── dt-aid.lua             # lighttable module: phase 1 + 2
    └── INSTALL.md             # install + usage guide
```

## Invariants to maintain

- `core/` imports nothing from `cli/`. The planned Lua plugin depends on this.
- Never mutate `library.db`. Reads use `?mode=ro` URIs; writes go to XMP sidecars.
- `tagging.is_managed_tag()` is the single source of truth for which tags we own. Any new auto-tag root must be added there AND covered by a test in `test_tagging_merge_preserves_user_tags`.
- `tagging.is_face_tag()` / `tagging.is_elements_tag()` partition the managed namespace by which detector produced each tag. `pipeline.scan()` uses them to carry over tags from a detector that was skipped by idempotency (so e.g. scanning elements-only doesn't wipe existing people tags). New tag categories owned by a new detector must add a matching classifier and be covered by a carry-over test.
- `embeddings.npy` and `embeddings.meta.parquet` must stay row-aligned. Always append via `EmbeddingStore.append()`.
- Every write path (XMP, state.db, embeddings) writes to a `.tmp` file and renames. No partial writes.

## Adding a new detector

1. Implement under `core/faces/` or `core/elements/` with a class that exposes a single method (`detect(image_path)` or `tag(image_path)`).
2. Produce `Tag` instances via the helpers in `core.tagging` — never construct raw tag strings.
3. Bump the relevant model-version constant in `core.config` (`FACES_MODEL_VERSION` / `ELEMENTS_MODEL_VERSION`) so existing `state.db` rows mark the image as stale and trigger re-processing.
4. Wire it into `core.pipeline.scan()`.
5. Add a unit test that stubs the detector and asserts the emitted tags.
6. Update `docs/architecture.md`, `docs/tag-namespace.md`, and `docs/CHANGELOG.md`.

## Adding a new output format

Currently only XMP is implemented. For the future direct-DB writer:
1. It must refuse to run if darktable is detected (already gated in `pipeline.scan`).
2. It must back up `library.db` before opening read-write.
3. It must be opt-in (`--write db`), never default.
4. Add integration tests that run against a copy of a real library.

## Code style

- `ruff` enforces formatting and linting (`pyproject.toml` configures it).
- Type hints are expected on public functions.
- Prefer dataclasses over dicts for structured returns.
- Docstrings only where behavior is non-obvious (matches repo-wide "no comments unless WHY is non-obvious" policy).

## Debugging

Enable verbose JSON logs on either frontend:

```bash
dt-aid -v scan --faces --dry-run 2>&1 | jq .
```

Useful filters:

```bash
# All skipped-file warnings
dt-aid -v scan --faces --dry-run 2>&1 | grep "image missing"

# RAW decode failures
dt-aid -v scan --faces --dry-run 2>&1 | grep "failed to decode"
```
