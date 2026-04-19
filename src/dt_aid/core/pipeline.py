from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import tagging, xmp
from .config import ELEMENTS_MODEL_VERSION, FACES_MODEL_VERSION, Settings
from .darktable_db import DarktableRunningError, DtImage, connect_readonly, is_darktable_running, iter_images
from .device import resolve_onnx_providers, resolve_torch_device
from .faces.detector import FaceDetector
from .faces.embeddings import EmbeddingStore, FaceRow, ReferenceLibrary
from .faces.matcher import FaceMatcher
from .state import mark_processed, needs_processing, open_state
from .tagging import Tag

log = logging.getLogger(__name__)


@dataclass
class ScanOptions:
    do_faces: bool = True
    do_elements: bool = False
    write_mode: str = "xmp"  # "xmp" | "db" | "none"
    dry_run: bool = True
    force: bool = False
    path_prefix: Path | None = None


@dataclass
class ImageResult:
    image: DtImage
    new_tags: list[Tag] = field(default_factory=list)
    final_sidecar_tags: list[str] | None = None  # None if dry-run or no writes
    ran_faces: bool = False
    ran_elements: bool = False


@dataclass
class ScanReport:
    total: int = 0
    processed: int = 0
    skipped: int = 0
    faces_detected: int = 0
    elements_tagged: int = 0
    written: int = 0


ProgressFn = Callable[[str, int, int], None]


def _noop_progress(_msg: str, _current: int, _total: int) -> None:
    return


def scan(
    settings: Settings,
    options: ScanOptions,
    *,
    progress: ProgressFn | None = None,
) -> tuple[ScanReport, list[ImageResult]]:
    """
    End-to-end scan: iterate darktable images, run detectors, write tags.
    Returns a report and per-image results. The core is frontend-agnostic
    (no CLI imports); pass a progress callback to surface progress.

    write_mode:
        "xmp"  : write to <image>.xmp sidecars (safe)
        "db"   : not implemented yet; raises NotImplementedError
        "none" : read-only, do not write anything
    """
    progress = progress or _noop_progress

    if is_darktable_running():
        raise DarktableRunningError(
            "darktable is running. Close it or run against a copy of library.db."
        )

    settings.ensure_dirs()

    if options.write_mode == "db":
        raise NotImplementedError(
            "Direct DB writes are not yet implemented. Use --write xmp."
        )

    report = ScanReport()
    results: list[ImageResult] = []

    detector, matcher, store = (None, None, None)
    if options.do_faces:
        providers = resolve_onnx_providers(settings.device)
        detector = FaceDetector(
            models_dir=settings.models_dir,
            det_size=settings.face_det_size,
            det_score_threshold=settings.face_det_score_threshold,
            providers=providers,
        )
        refs = ReferenceLibrary(settings.face_references_dir).load_all()
        matcher = FaceMatcher(refs, threshold=settings.face_match_threshold)
        store = EmbeddingStore(settings.face_embeddings_npy, settings.face_embeddings_meta)

    clip_tagger = None
    if options.do_elements:
        from .elements.clip_tagger import ClipTagger
        from .elements.labels import default_label_set, load_label_set

        label_set = (
            load_label_set(settings.elements_labels_file)
            if settings.elements_labels_file
            else default_label_set()
        )
        clip_tagger = ClipTagger(
            label_set=label_set,
            cache_dir=settings.models_dir,
            threshold=settings.elements_threshold,
            device=resolve_torch_device(settings.device),
            batch_size=settings.elements_batch_size,
        )

    dt_conn = connect_readonly(settings.darktable_library)
    try:
        images: Iterable[DtImage] = list(
            iter_images(dt_conn, path_prefix=options.path_prefix)
        )
    finally:
        dt_conn.close()

    report.total = len(images)

    # --force re-detects faces for every image with a known-good path. The
    # embedding store is append-only, so without pruning first we'd double
    # the cache. Delete stale rows for the soon-to-be-reprocessed images
    # in a single atomic pass.
    if (
        options.do_faces
        and options.force
        and not options.dry_run
        and store is not None
    ):
        reprocess_paths = {str(img.path) for img in images if img.path.exists()}
        if reprocess_paths:
            removed = store.delete_rows_for_images(reprocess_paths)
            log.info(
                "force: removed %d stale embeddings across %d images",
                removed,
                len(reprocess_paths),
            )

    with open_state(settings.state_db) as state:
        # Images awaiting batched element tagging. Each entry references
        # the same ImageResult that's already been appended to `results`,
        # so commits fill in element tags + final_sidecar_tags in place
        # and iteration order is preserved.
        pending: list[tuple[DtImage, ImageResult, bool, bool]] = []

        def _commit(image: DtImage, result: ImageResult, rf: bool, re: bool) -> None:
            new_tags = result.new_tags
            if new_tags and not options.dry_run and options.write_mode == "xmp":
                existing_flat, existing_hier = xmp.read_subjects(image.path)
                base = existing_hier or image.tags
                merged_hier = tagging.merge_managed(base, new_tags)
                # Carry over tags from any detector that was skipped this
                # pass (merge_managed drops every managed tag), so e.g.
                # scanning --elements on a faces-tagged library doesn't
                # wipe the people|* tags.
                if not rf:
                    for t in base:
                        if tagging.is_face_tag(t) and t not in merged_hier:
                            merged_hier.append(t)
                if not re:
                    for t in base:
                        if tagging.is_elements_tag(t) and t not in merged_hier:
                            merged_hier.append(t)
                leaves = [xmp.leaf_label(t) for t in merged_hier]
                seen: set[str] = set()
                flat: list[str] = []
                for leaf in (*existing_flat, *leaves):
                    if leaf not in seen:
                        seen.add(leaf)
                        flat.append(leaf)
                xmp.write_subjects(
                    image.path,
                    flat_tags=flat,
                    hierarchical_tags=merged_hier,
                )
                result.final_sidecar_tags = merged_hier
                report.written += 1

            if not options.dry_run:
                mark_processed(
                    state,
                    str(image.path),
                    dt_image_id=image.id,
                    faces_version=FACES_MODEL_VERSION if rf else None,
                    elements_version=ELEMENTS_MODEL_VERSION if re else None,
                )
            report.processed += 1

        def _flush_elements() -> None:
            if not pending:
                return
            if clip_tagger is None:
                for image, result, rf, re in pending:
                    _commit(image, result, rf, re)
                pending.clear()
                return
            paths = [entry[0].path for entry in pending]
            all_detections = clip_tagger.tag_batch(paths)
            for (image, result, rf, re), detections in zip(pending, all_detections):
                for d in detections:
                    if d.kind == "object":
                        result.new_tags.append(tagging.object_tag(d.label))
                    elif d.kind == "scene":
                        result.new_tags.append(tagging.scene_tag(d.label))
                    elif d.kind == "attr":
                        result.new_tags.append(tagging.attr_tag(d.label))
                if detections:
                    result.new_tags.append(tagging.elements_provenance_tag())
                report.elements_tagged += len(detections)
                _commit(image, result, rf, re)
            pending.clear()

        for idx, image in enumerate(images, start=1):
            progress("scan", idx, report.total)

            if not image.path.exists():
                log.warning("image missing on disk: %s", image.path)
                report.skipped += 1
                continue

            run_faces, run_elements = needs_processing(
                state,
                str(image.path),
                faces_version=FACES_MODEL_VERSION if options.do_faces else None,
                elements_version=ELEMENTS_MODEL_VERSION if options.do_elements else None,
                force=options.force,
            )
            if not run_faces and not run_elements:
                report.skipped += 1
                continue

            new_tags: list[Tag] = []

            if run_faces and detector is not None:
                detections = detector.detect(image.path)
                face_rows: list[FaceRow] = []
                for det in detections:
                    match = matcher.match(det.embedding) if matcher else None
                    if match is not None:
                        new_tags.append(tagging.person_tag(match.name))
                        face_rows.append(
                            FaceRow(
                                image_path=str(image.path),
                                dt_image_id=image.id,
                                bbox=det.bbox,
                                det_score=det.det_score,
                                embedding=det.embedding,
                                cluster_id=-2,  # matched to known person
                                label=match.name,
                            )
                        )
                    else:
                        face_rows.append(
                            FaceRow(
                                image_path=str(image.path),
                                dt_image_id=image.id,
                                bbox=det.bbox,
                                det_score=det.det_score,
                                embedding=det.embedding,
                                cluster_id=-1,
                                label="",
                            )
                        )
                if detections:
                    new_tags.append(tagging.faces_provenance_tag())
                report.faces_detected += len(detections)
                if not options.dry_run and store is not None and face_rows:
                    store.append(face_rows)

            result = ImageResult(
                image=image,
                new_tags=new_tags,
                ran_faces=run_faces,
                ran_elements=run_elements,
            )
            results.append(result)

            if run_elements and clip_tagger is not None:
                pending.append((image, result, run_faces, run_elements))
                if len(pending) >= settings.elements_batch_size:
                    _flush_elements()
            else:
                _commit(image, result, run_faces, run_elements)

        _flush_elements()

    return report, results
