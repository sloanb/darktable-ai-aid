from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import FACES_MODEL_NAME
from ..image_io import load_bgr


@dataclass
class FaceDetection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    det_score: float
    embedding: np.ndarray  # float32 [512], L2-normalized


class FaceDetector:
    """Wraps InsightFace buffalo_l (RetinaFace detector + ArcFace embeddings)."""

    def __init__(
        self,
        *,
        models_dir: Path,
        det_size: int = 640,
        det_score_threshold: float = 0.5,
        providers: list[str] | None = None,
    ) -> None:
        from insightface.app import FaceAnalysis

        models_dir.mkdir(parents=True, exist_ok=True)
        if providers is None:
            providers = ["CPUExecutionProvider"]
        # ctx_id=0 selects GPU 0 when a CUDA provider is present; ONNX Runtime
        # ignores it for CPU-only provider lists.
        ctx_id = 0 if any(p.startswith("CUDA") for p in providers) else -1
        self._app = FaceAnalysis(
            name=FACES_MODEL_NAME,
            root=str(models_dir),
            providers=providers,
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self._threshold = det_score_threshold

    def detect(self, image_path: Path) -> list[FaceDetection]:
        img = load_bgr(image_path)
        if img is None:
            return []
        return self.detect_array(img)

    def detect_array(self, bgr_image: np.ndarray) -> list[FaceDetection]:
        faces = self._app.get(bgr_image)
        out: list[FaceDetection] = []
        for f in faces:
            if float(f.det_score) < self._threshold:
                continue
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            # guard: normed_embedding is already unit-length, but be defensive
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
            x1, y1, x2, y2 = (float(v) for v in f.bbox)
            out.append(
                FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    det_score=float(f.det_score),
                    embedding=emb,
                )
            )
        return out
