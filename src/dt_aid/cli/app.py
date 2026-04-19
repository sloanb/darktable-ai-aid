from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

from ..core.config import load_settings
from ..core.darktable_db import DarktableRunningError
from ..core.elements.clip_tagger import MissingElementsExtraError
from ..core.faces.embeddings import EmbeddingStore, ReferenceLibrary
from ..core.pipeline import ScanOptions, scan
from ..core.state import open_state, promote_cluster
from .logging_setup import setup_logging
from .progress import CliProgress

log = logging.getLogger("dt_aid")
console = Console(stderr=False)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dt-aid",
        description="AI-assisted auto-tagging for darktable libraries.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    scan_p = sub.add_parser("scan", help="Scan library and write tags.")
    scan_p.add_argument("--path", type=Path, default=None, help="Only images under this path")
    scan_p.add_argument("--faces", action="store_true", help="Run face detection + matching")
    scan_p.add_argument("--elements", action="store_true", help="Run CLIP element tagger")
    scan_p.add_argument(
        "--write",
        choices=["xmp", "db", "none"],
        default="xmp",
        help="Where to write tags (default: xmp).",
    )
    scan_p.add_argument("--dry-run", action="store_true", help="Do not write anything")
    scan_p.add_argument("--force", action="store_true", help="Re-process already-seen images")
    scan_p.add_argument(
        "--library",
        type=Path,
        default=None,
        help="Path to darktable library.db (overrides default)",
    )
    scan_p.add_argument(
        "--known-faces",
        type=Path,
        default=None,
        help="Directory of labeled reference photos (<name>/*.jpg)",
    )
    scan_p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Inference device (default: auto — CUDA if available else CPU)",
    )

    faces_p = sub.add_parser("faces", help="Manage face clusters and references.")
    faces_sub = faces_p.add_subparsers(dest="faces_cmd", required=True)

    relabel_p = faces_sub.add_parser(
        "relabel", help="Promote a cluster to a named person."
    )
    relabel_p.add_argument("cluster_id", type=int)
    relabel_p.add_argument("name")

    build_refs_p = faces_sub.add_parser(
        "build-refs",
        help="Build reference embeddings from labeled photos in a directory.",
    )
    build_refs_p.add_argument("known_faces", type=Path)
    build_refs_p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Inference device (default: auto — CUDA if available else CPU)",
    )

    list_p = faces_sub.add_parser(
        "list",
        help="Print clusters and references as JSON (for tools/plugins).",
    )
    list_p.add_argument(
        "--min-size",
        type=int,
        default=1,
        help="Only include clusters with at least N faces (default: 1).",
    )

    add_image_p = faces_sub.add_parser(
        "add-image",
        help="Teach dt-aid that a face in a specific image belongs to a named person.",
    )
    add_image_p.add_argument("image", type=Path)
    add_image_p.add_argument("name")
    add_image_p.add_argument(
        "--face-index",
        type=int,
        default=None,
        help="Which detected face to use (0-based). Defaults to the largest face.",
    )
    add_image_p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Inference device (default: auto — CUDA if available else CPU)",
    )
    add_image_p.add_argument(
        "--json",
        action="store_true",
        help="Emit a one-line JSON summary on stdout (for plugin/tooling use).",
    )

    rematch_p = faces_sub.add_parser(
        "rematch",
        help="Re-match unlabeled face embeddings against current references (no re-detection).",
    )
    rematch_p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold (default: DT_AID_FACE_MATCH_THRESHOLD = 0.5).",
    )
    rematch_p.add_argument(
        "--no-write-xmp",
        action="store_true",
        help="Update parquet but do not rewrite XMP sidecars.",
    )

    cluster_p = faces_sub.add_parser(
        "cluster",
        help="Cluster unmatched face embeddings and tag affected XMPs with people|unknown|cluster-NNN.",
    )
    cluster_p.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN min_cluster_size (default: 5). Lower = more, smaller clusters.",
    )
    cluster_p.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples. Defaults to min_cluster_size.",
    )
    cluster_p.add_argument(
        "--no-write-xmp",
        action="store_true",
        help="Update the parquet/state.db but do not rewrite XMP sidecars.",
    )

    return p


def _settings_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if getattr(args, "library", None):
        overrides["darktable_library"] = args.library
    if getattr(args, "known_faces", None):
        overrides["known_faces_dir"] = args.known_faces
    if getattr(args, "device", None):
        overrides["device"] = args.device
    return overrides


def cmd_scan(args: argparse.Namespace) -> int:
    settings = load_settings(**_settings_overrides(args))
    options = ScanOptions(
        do_faces=args.faces,
        do_elements=args.elements,
        write_mode=args.write,
        dry_run=args.dry_run,
        force=args.force,
        path_prefix=args.path,
    )
    if not (options.do_faces or options.do_elements):
        console.print("[yellow]nothing to do: pass --faces and/or --elements[/yellow]")
        return 2

    try:
        with CliProgress("scan") as pg:
            report, results = scan(settings, options, progress=pg.update)
    except DarktableRunningError as e:
        console.print(f"[red]{e}[/red]")
        return 3
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return 4
    except MissingElementsExtraError as e:
        console.print(f"[red]{e}[/red]")
        return 5

    console.print(
        f"[green]done[/green]: total={report.total} processed={report.processed} "
        f"skipped={report.skipped} faces={report.faces_detected} "
        f"elements={report.elements_tagged} written={report.written}"
    )
    if options.dry_run:
        console.print("[yellow]dry-run: no files written[/yellow]")
        tagged = [r for r in results if r.new_tags]
        console.print(f"images with detections: {len(tagged)}")
        for r in tagged[:10]:
            console.print(f"  {r.image.path}")
            for t in r.new_tags:
                console.print(f"    + {t.value}")
    return 0


def cmd_faces_relabel(args: argparse.Namespace) -> int:
    settings = load_settings()
    settings.ensure_dirs()
    store = EmbeddingStore(settings.face_embeddings_npy, settings.face_embeddings_meta)

    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from ..core.xmp_sync import sync_xmp_for_images

    if not settings.face_embeddings_meta.exists():
        console.print("[red]no face embeddings found. Run `dt-aid scan --faces` first.[/red]")
        return 1

    table = pq.read_table(settings.face_embeddings_meta)
    matching = table.filter(pc.equal(table["cluster_id"], args.cluster_id))
    if matching.num_rows == 0:
        console.print(f"[red]cluster {args.cluster_id} has no members[/red]")
        return 1

    # Pull embeddings for this cluster, save as a new reference library entry.
    all_vecs = np.load(settings.face_embeddings_npy, mmap_mode="r")
    row_ids_np = np.asarray(matching["row"].to_numpy(), dtype=np.int64)
    cluster_vecs = np.ascontiguousarray(all_vecs[row_ids_np])
    refs = ReferenceLibrary(settings.face_references_dir)
    refs.save(args.name, cluster_vecs)

    # Update parquet: set label=name AND cluster_id=-2 (matched known person)
    # so these rows are indistinguishable from freshly-matched faces.
    row_ids = [int(r) for r in matching["row"].to_pylist()]
    store.update_assignments(
        labels={r: args.name for r in row_ids},
        cluster_ids={r: -2 for r in row_ids},
    )
    with open_state(settings.state_db) as state:
        promote_cluster(state, args.cluster_id, args.name)

    # Re-sync XMPs for every image touched by this cluster so the
    # people|unknown|cluster-NNN tag is replaced by people|<name>.
    image_paths = set(matching["image_path"].to_pylist())
    with open_state(settings.state_db) as state:
        with CliProgress("xmp_sync") as pg:
            written = sync_xmp_for_images(
                image_paths, store=store, state_conn=state, progress=pg.update
            )

    console.print(
        f"[green]promoted[/green] cluster {args.cluster_id} -> '{args.name}' "
        f"({matching.num_rows} faces across {len(image_paths)} images, "
        f"{written} XMPs rewritten)"
    )
    return 0


def cmd_faces_build_refs(args: argparse.Namespace) -> int:
    settings = load_settings(**_settings_overrides(args))
    settings.ensure_dirs()
    from ..core.device import resolve_onnx_providers
    from ..core.faces.detector import FaceDetector

    detector = FaceDetector(
        models_dir=settings.models_dir,
        det_size=settings.face_det_size,
        det_score_threshold=settings.face_det_score_threshold,
        providers=resolve_onnx_providers(settings.device),
    )
    refs = ReferenceLibrary(settings.face_references_dir)

    root = args.known_faces
    if not root.is_dir():
        console.print(f"[red]not a directory: {root}[/red]")
        return 1

    import numpy as np

    built = 0
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        vectors: list[np.ndarray] = []
        for img in sorted(person_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            dets = detector.detect(img)
            if not dets:
                log.warning("no face found in reference %s", img)
                continue
            # pick the largest face in each reference photo
            best = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
            vectors.append(best.embedding)
        if not vectors:
            console.print(f"[yellow]no usable faces for {person_dir.name}[/yellow]")
            continue
        refs.save(person_dir.name, np.stack(vectors))
        built += 1
        console.print(f"[green]{person_dir.name}[/green]: {len(vectors)} references")

    console.print(f"[green]built references for {built} people[/green]")
    return 0


def cmd_faces_list(args: argparse.Namespace) -> int:
    """Emit JSON: references + clusters. Always JSON (it's a tooling command)."""
    import json as _json
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from ..core.faces.embeddings import ReferenceLibrary
    from ..core.state import open_state

    settings = load_settings(**_settings_overrides(args))
    settings.ensure_dirs()

    refs_lib = ReferenceLibrary(settings.face_references_dir)
    references = []
    for name in refs_lib.names():
        vecs = refs_lib.load(name)
        references.append({"name": name, "vector_count": int(vecs.shape[0])})

    clusters: list[dict] = []
    if settings.face_embeddings_meta.exists():
        table = pq.read_table(settings.face_embeddings_meta)
        cluster_ids = table["cluster_id"].to_pylist()
        paths = table["image_path"].to_pylist()
        counts: dict[int, int] = {}
        distinct_images: dict[int, set[str]] = {}
        for cid, path in zip(cluster_ids, paths):
            if cid is None or int(cid) < 0:
                continue
            c = int(cid)
            counts[c] = counts.get(c, 0) + 1
            distinct_images.setdefault(c, set()).add(path)

        with open_state(settings.state_db) as state:
            rows = state.execute(
                "SELECT cluster_id, label FROM clusters"
            ).fetchall()
            label_for: dict[int, str | None] = {
                int(r["cluster_id"]): r["label"] for r in rows
            }

        for cid, n in sorted(counts.items(), key=lambda x: -x[1]):
            if n < args.min_size:
                continue
            clusters.append({
                "cluster_id": cid,
                "face_count": n,
                "image_count": len(distinct_images.get(cid, ())),
                "label": label_for.get(cid),
            })

    payload = {"references": references, "clusters": clusters}
    print(_json.dumps(payload))
    return 0


def cmd_faces_add_image(args: argparse.Namespace) -> int:
    import json as _json

    from ..core.device import resolve_onnx_providers
    from ..core.faces.add_image_runner import run_add_image

    settings = load_settings(**_settings_overrides(args))
    try:
        report = run_add_image(
            settings,
            image_path=args.image,
            name=args.name,
            face_index=args.face_index,
            providers=resolve_onnx_providers(settings.device),
        )
    except (FileNotFoundError, ValueError, IndexError) as e:
        if args.json:
            print(_json.dumps({"ok": False, "error": str(e)}))
        else:
            console.print(f"[red]{e}[/red]")
        return 1

    if args.json:
        print(_json.dumps({
            "ok": True,
            "image_path": report.image_path,
            "faces_detected": report.faces_detected,
            "chosen_face_index": report.chosen_face_index,
            "reference_count_after": report.reference_count_after,
            "parquet_row_updated": report.parquet_row_updated,
            "xmp_written": report.xmp_written,
            "name": args.name,
        }))
    else:
        console.print(
            f"[green]added[/green]: {args.name} now has {report.reference_count_after} "
            f"reference vectors (used face {report.chosen_face_index} of {report.faces_detected})"
        )
        if report.parquet_row_updated:
            console.print("  [green]✓[/green] matching parquet row relabeled")
        if report.xmp_written:
            console.print("  [green]✓[/green] XMP sidecar rewritten")
        if not report.parquet_row_updated:
            console.print(
                "  [yellow]note[/yellow]: image not in embedding store yet; "
                "run `dt-aid scan --faces` to include it."
            )
    return 0


def cmd_faces_rematch(args: argparse.Namespace) -> int:
    from ..core.faces.rematch_runner import run_rematch

    settings = load_settings(**_settings_overrides(args))
    with CliProgress("xmp_sync") as pg:
        report = run_rematch(
            settings,
            threshold=args.threshold,
            write_xmp=not args.no_write_xmp,
            progress=pg.update,
        )
    console.print(
        f"[green]rematch done[/green]: candidates={report.candidates} "
        f"new_matches={report.new_matches} xmps_written={report.xmps_written}"
    )
    if report.by_person:
        console.print("per-person:")
        for name, n in sorted(report.by_person.items(), key=lambda x: -x[1]):
            console.print(f"  {name}: {n}")
    return 0


def cmd_faces_cluster(args: argparse.Namespace) -> int:
    from ..core.faces.cluster_runner import run_cluster

    settings = load_settings(**_settings_overrides(args))
    with CliProgress("xmp_sync") as pg:
        report = run_cluster(
            settings,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            write_xmp=not args.no_write_xmp,
            progress=pg.update,
        )
    console.print(
        f"[green]cluster done[/green]: unmatched={report.unmatched_total} "
        f"clustered={report.clustered} noise={report.noise} "
        f"new_clusters={report.new_clusters} xmps_written={report.xmps_written}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)
    if args.cmd == "scan":
        return cmd_scan(args)
    if args.cmd == "faces":
        if args.faces_cmd == "relabel":
            return cmd_faces_relabel(args)
        if args.faces_cmd == "build-refs":
            return cmd_faces_build_refs(args)
        if args.faces_cmd == "cluster":
            return cmd_faces_cluster(args)
        if args.faces_cmd == "rematch":
            return cmd_faces_rematch(args)
        if args.faces_cmd == "add-image":
            return cmd_faces_add_image(args)
        if args.faces_cmd == "list":
            return cmd_faces_list(args)
    parser.error(f"unknown command: {args.cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
