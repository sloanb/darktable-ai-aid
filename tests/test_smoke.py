from __future__ import annotations

from pathlib import Path

import pytest

import numpy as np

from dt_aid.core import tagging, xmp
from dt_aid.core.darktable_db import connect_readonly, count_images, iter_images
from dt_aid.core.faces.embeddings import EmbeddingStore, FaceRow
from dt_aid.core.state import (
    mark_processed,
    needs_processing,
    open_state,
    promote_cluster,
)
from dt_aid.core.tagging import Tag, TagKind

from .fixtures.make_library import build as build_fixture


@pytest.fixture()
def fixture_lib(tmp_path: Path) -> tuple[Path, list[Path]]:
    return build_fixture(tmp_path / "lib")


def test_darktable_db_reads_fixture(fixture_lib):
    db_path, image_paths = fixture_lib
    conn = connect_readonly(db_path)
    try:
        assert count_images(conn) == 3
        listed = list(iter_images(conn))
        assert len(listed) == 3
        assert {img.path for img in listed} == set(image_paths)
        # first image had a user tag "keepers"
        first = next(img for img in listed if img.id == 1)
        assert "keepers" in first.tags
    finally:
        conn.close()


def test_tagging_namespace_and_slug():
    assert tagging.person_tag("Alice Smith").value == "people|alice-smith"
    assert tagging.object_tag("Dining Table").value == "auto|object|dining-table"
    assert tagging.cluster_tag(7).value == "people|unknown|cluster-007"
    assert tagging.is_managed_tag("people|alice")
    assert tagging.is_managed_tag("auto|object|dog")
    assert not tagging.is_managed_tag("keepers")
    assert not tagging.is_managed_tag("vacation|italy")


def test_tagging_merge_preserves_user_tags():
    existing = ["keepers", "vacation|italy", "people|old-label", "auto|object|stale"]
    new = [
        Tag(TagKind.PERSON, "people|alice"),
        Tag(TagKind.OBJECT, "auto|object|dog"),
    ]
    merged = tagging.merge_managed(existing, new)
    assert "keepers" in merged
    assert "vacation|italy" in merged
    assert "people|old-label" not in merged  # managed, replaced
    assert "auto|object|stale" not in merged
    assert "people|alice" in merged
    assert "auto|object|dog" in merged


def test_tagging_hierarchical_ancestors():
    assert tagging.hierarchical_ancestors("people|family|alice") == [
        "people",
        "people|family",
        "people|family|alice",
    ]


def test_xmp_roundtrip(tmp_path: Path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff")  # fake JPEG header; we never decode it
    sidecar = xmp.write_subjects(
        img,
        flat_tags=["alice", "dog"],
        hierarchical_tags=["people|alice", "auto|object|dog"],
    )
    assert sidecar.exists()

    flat, hier = xmp.read_subjects(img)
    assert "alice" in flat and "dog" in flat
    # ancestors expanded
    assert "people" in hier
    assert "people|alice" in hier
    assert "auto" in hier
    assert "auto|object" in hier
    assert "auto|object|dog" in hier


def test_embedding_store_atomic_append(tmp_path: Path):
    store = EmbeddingStore(
        npy_path=tmp_path / "embeddings.npy",
        meta_path=tmp_path / "embeddings.meta.parquet",
    )
    rows = [
        FaceRow(
            image_path="/x/a.jpg",
            dt_image_id=1,
            bbox=(0.0, 0.0, 10.0, 10.0),
            det_score=0.9,
            embedding=np.ones(512, dtype=np.float32) / np.sqrt(512),
        )
    ]
    store.append(rows)
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "embeddings.meta.parquet").exists()
    # No stale tmp files — regression guard for the .npy.tmp.npy bug
    assert not (tmp_path / "embeddings.npy.tmp").exists()
    assert not (tmp_path / "embeddings.npy.tmp.npy").exists()
    loaded = np.load(tmp_path / "embeddings.npy")
    assert loaded.shape == (1, 512)

    # Appending again should extend the matrix, not overwrite
    store.append(rows)
    loaded = np.load(tmp_path / "embeddings.npy")
    assert loaded.shape == (2, 512)


def test_reference_library_append_merges(tmp_path: Path):
    from dt_aid.core.faces.embeddings import ReferenceLibrary

    lib = ReferenceLibrary(tmp_path)
    v1 = np.zeros((2, 512), dtype=np.float32); v1[:, 0] = 1.0
    v2 = np.zeros((3, 512), dtype=np.float32); v2[:, 100] = 1.0

    # First append creates the file
    n = lib.append("alice", v1)
    assert n == 2
    loaded = lib.load("alice")
    assert loaded.shape == (2, 512)
    # L2-normalized
    norms = np.linalg.norm(loaded, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # Second append merges (does not replace)
    n = lib.append("alice", v2)
    assert n == 5
    loaded = lib.load("alice")
    assert loaded.shape == (5, 512)
    # No stray .tmp files
    assert not (tmp_path / "alice.npy.tmp").exists()

    # save() still replaces
    lib.save("alice", np.zeros((1, 512), dtype=np.float32) + np.eye(1, 512))
    assert lib.load("alice").shape == (1, 512)


def test_embedding_store_delete_rows_for_images(tmp_path: Path):
    import pyarrow.parquet as pq

    store = EmbeddingStore(
        npy_path=tmp_path / "embeddings.npy",
        meta_path=tmp_path / "embeddings.meta.parquet",
    )
    rng = np.random.default_rng(0)

    def _row(path: str) -> FaceRow:
        v = rng.standard_normal(512).astype(np.float32)
        v /= np.linalg.norm(v)
        return FaceRow(
            image_path=path,
            dt_image_id=0,
            bbox=(0, 0, 1, 1),
            det_score=0.9,
            embedding=v,
        )

    # Seed 5 faces across 3 images (a has 2, b has 2, c has 1)
    store.append([_row("/x/a.jpg"), _row("/x/a.jpg"), _row("/x/b.jpg"), _row("/x/b.jpg"), _row("/x/c.jpg")])
    assert np.load(tmp_path / "embeddings.npy").shape == (5, 512)

    # Delete rows for images a and b
    removed = store.delete_rows_for_images({"/x/a.jpg", "/x/b.jpg"})
    assert removed == 4

    # Only c's row remains
    vecs = np.load(tmp_path / "embeddings.npy")
    assert vecs.shape == (1, 512)
    meta = pq.read_table(tmp_path / "embeddings.meta.parquet")
    assert meta.num_rows == 1
    assert meta.column("image_path").to_pylist() == ["/x/c.jpg"]
    # row column must be renumbered to match new npy positions (0..N-1)
    assert meta.column("row").to_pylist() == [0]

    # Deleting a non-existent image is a no-op
    assert store.delete_rows_for_images({"/x/missing.jpg"}) == 0
    assert np.load(tmp_path / "embeddings.npy").shape == (1, 512)

    # After delete + append, the new rows continue from the current end
    store.append([_row("/x/d.jpg")])
    vecs = np.load(tmp_path / "embeddings.npy")
    meta = pq.read_table(tmp_path / "embeddings.meta.parquet")
    assert vecs.shape == (2, 512)
    assert meta.column("row").to_pylist() == [0, 1]
    assert meta.column("image_path").to_pylist() == ["/x/c.jpg", "/x/d.jpg"]


def test_cluster_unknowns_separates_well_separated_groups():
    from dt_aid.core.faces.cluster import cluster_unknowns

    rng = np.random.default_rng(42)

    def _blob(center: np.ndarray, n: int, jitter: float = 0.02) -> np.ndarray:
        v = np.tile(center, (n, 1)) + rng.standard_normal((n, 512)) * jitter
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v.astype(np.float32)

    # Two well-separated blobs, 10 each, in 512-D. The two centers are
    # near-orthogonal so cosine / euclidean distance between blobs is large.
    c1 = np.zeros(512, dtype=np.float32); c1[0] = 1.0
    c2 = np.zeros(512, dtype=np.float32); c2[100] = 1.0
    embeddings = np.concatenate([_blob(c1, 10), _blob(c2, 10)], axis=0)

    labels = cluster_unknowns(embeddings, min_cluster_size=3)
    assigned = labels[labels >= 0]
    # At least two non-noise clusters should emerge.
    assert len(set(assigned.tolist())) >= 2
    # And each blob should be mostly in one cluster.
    blob1_labels = labels[:10]
    blob2_labels = labels[10:]
    # Dominant label per blob should cover the majority of its points.
    from collections import Counter
    c1_top = Counter(l for l in blob1_labels if l >= 0).most_common(1)
    c2_top = Counter(l for l in blob2_labels if l >= 0).most_common(1)
    assert c1_top and c1_top[0][1] >= 7
    assert c2_top and c2_top[0][1] >= 7
    # The two blobs should not share their dominant label.
    assert c1_top[0][0] != c2_top[0][0]


def test_xmp_sync_writes_cluster_tag(tmp_path: Path):
    from dt_aid.core.faces.embeddings import EmbeddingStore, FaceRow
    from dt_aid.core.state import open_state, upsert_cluster
    from dt_aid.core.xmp_sync import sync_xmp_for_images

    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff")  # fake JPEG header; we never decode it

    store = EmbeddingStore(
        npy_path=tmp_path / "embeddings.npy",
        meta_path=tmp_path / "embeddings.meta.parquet",
    )
    vec = np.ones(512, dtype=np.float32) / np.sqrt(512)
    # one matched face (parker) + one clustered unknown (cluster 3)
    store.append([
        FaceRow(
            image_path=str(image),
            dt_image_id=1,
            bbox=(0, 0, 10, 10),
            det_score=0.9,
            embedding=vec,
            cluster_id=-2,
            label="parker",
        ),
        FaceRow(
            image_path=str(image),
            dt_image_id=1,
            bbox=(20, 20, 30, 30),
            det_score=0.8,
            embedding=vec,
            cluster_id=3,
            label="",
        ),
    ])

    state_db = tmp_path / "state.db"
    with open_state(state_db) as state:
        upsert_cluster(state, 3)
        written = sync_xmp_for_images(
            {str(image)}, store=store, state_conn=state
        )

    assert written == 1
    flat, hier = __import__("dt_aid.core.xmp", fromlist=["read_subjects"]).read_subjects(image)
    assert "people|parker" in hier
    assert "people|unknown|cluster-003" in hier
    # Provenance tag should be present
    assert any(t.startswith("auto|_meta|model-faces-") for t in hier)


def test_rematch_runner_promotes_unlabeled_faces(tmp_path: Path, monkeypatch):
    """
    Seed an embedding store with (a) a real-world "noise" face that is
    semantically close to a reference vector, (b) an unrelated face that
    should not match, plus a reference library containing one person.
    Verify rematch labels only the close face and re-syncs its XMP.
    """
    from dt_aid.core.config import Settings
    from dt_aid.core.faces.embeddings import EmbeddingStore, FaceRow, ReferenceLibrary
    from dt_aid.core.faces.rematch_runner import run_rematch

    # 1) Build a Settings that points at tmp_path
    data_dir = tmp_path / "data"
    settings = Settings(
        darktable_library=tmp_path / "library.db",  # never opened
        data_dir=data_dir,
        face_match_threshold=0.5,
    )
    settings.ensure_dirs()

    # 2) Reference library: one person "alice" with a distinctive vector
    alice_vec = np.zeros(512, dtype=np.float32)
    alice_vec[0] = 1.0  # canonical axis-1 unit vector
    refs = ReferenceLibrary(settings.face_references_dir)
    refs.save("alice", alice_vec[None, :])

    # 3) Two images, each with one face
    img_close = tmp_path / "close.jpg"
    img_far = tmp_path / "far.jpg"
    img_close.write_bytes(b"\xff\xd8\xff")
    img_far.write_bytes(b"\xff\xd8\xff")

    # near-alice face: axis-1 with tiny noise -> cosine sim ~1.0
    close_vec = alice_vec.copy()
    close_vec[1] = 0.05
    close_vec /= np.linalg.norm(close_vec)

    # unrelated face: orthogonal direction
    far_vec = np.zeros(512, dtype=np.float32)
    far_vec[200] = 1.0

    store = EmbeddingStore(settings.face_embeddings_npy, settings.face_embeddings_meta)
    store.append([
        FaceRow(
            image_path=str(img_close),
            dt_image_id=1,
            bbox=(0, 0, 10, 10),
            det_score=0.9,
            embedding=close_vec.astype(np.float32),
            cluster_id=-1,
            label="",
        ),
        FaceRow(
            image_path=str(img_far),
            dt_image_id=2,
            bbox=(0, 0, 10, 10),
            det_score=0.9,
            embedding=far_vec,
            cluster_id=-1,
            label="",
        ),
    ])

    # 4) Run rematch
    report = run_rematch(settings, write_xmp=True)

    assert report.candidates == 2
    assert report.new_matches == 1
    assert report.by_person == {"alice": 1}
    assert report.xmps_written == 1

    # Parquet state: close.jpg now labeled, far.jpg still unlabeled
    import pyarrow.parquet as pq
    table = pq.read_table(settings.face_embeddings_meta)
    rows = {
        (p, l, c)
        for p, l, c in zip(
            table["image_path"].to_pylist(),
            table["label"].to_pylist(),
            table["cluster_id"].to_pylist(),
        )
    }
    assert (str(img_close), "alice", -2) in rows
    assert (str(img_far), "", -1) in rows

    # XMP for close.jpg should contain people|alice
    from dt_aid.core.xmp import read_subjects
    flat_close, hier_close = read_subjects(img_close)
    assert "people|alice" in hier_close

    # XMP for far.jpg should NOT have been written
    from dt_aid.core.xmp import sidecar_path
    assert not sidecar_path(img_far).exists()


def test_state_lifecycle(tmp_path: Path):
    db = tmp_path / "state.db"
    with open_state(db) as conn:
        run_faces, run_elements = needs_processing(
            conn, "/x/y.jpg", faces_version="F1", elements_version=None
        )
        assert run_faces is True
        assert run_elements is False

        mark_processed(
            conn, "/x/y.jpg", dt_image_id=42, faces_version="F1", elements_version=None
        )
        run_faces2, _ = needs_processing(
            conn, "/x/y.jpg", faces_version="F1", elements_version=None
        )
        assert run_faces2 is False

        # model upgrade => should re-run
        run_faces3, _ = needs_processing(
            conn, "/x/y.jpg", faces_version="F2", elements_version=None
        )
        assert run_faces3 is True

        promote_cluster(conn, 5, "alice")
        row = conn.execute(
            "SELECT label FROM clusters WHERE cluster_id = 5"
        ).fetchone()
        assert row["label"] == "alice"
