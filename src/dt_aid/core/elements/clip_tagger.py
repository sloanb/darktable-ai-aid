from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import CLIP_MODEL_NAME, CLIP_PRETRAINED
from ..image_io import load_rgb_pil
from .labels import LabelSet


class MissingElementsExtraError(RuntimeError):
    """Raised when the `[elements]` optional extra is not installed."""


@dataclass
class ElementDetection:
    kind: str  # "object" | "scene" | "attr"
    label: str
    score: float


class ClipTagger:
    """
    Zero-shot tagger using OpenCLIP. For each of the three label groups
    (objects, scenes, attrs) the tagger returns labels whose softmax-
    normalized cosine similarity against the image embedding exceeds
    `threshold`.

    The batch forward path (`tag_batch`) amortizes GPU kernel launch
    overhead — typical 2–4× speedup vs. per-image calls on CUDA. Text
    features are encoded once at init and cached as fp32 numpy arrays;
    `tag()` is a thin wrapper around `tag_batch([p])[0]`.
    """

    def __init__(
        self,
        *,
        label_set: LabelSet,
        cache_dir: Path,
        threshold: float = 0.25,
        device: str | None = None,
        batch_size: int = 16,
        use_fp16: bool | None = None,
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as e:
            raise MissingElementsExtraError(
                "element tagging requires the [elements] optional extra. Install with:\n"
                "  uv pip install -e '.[elements]'\n"
                "  (or: pip install -e '.[elements]')\n"
                "This pulls in torch + open-clip-torch (~2 GB). GPU users may want to "
                "install torch with CUDA first (see docs/development.md)."
            ) from e

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # fp16 defaults: on for CUDA (2× throughput, no meaningful accuracy
        # hit for zero-shot tagging), off for CPU (no speedup and some ops
        # lack fp16 kernels).
        if use_fp16 is None:
            use_fp16 = device != "cpu"

        self._device = device
        self._threshold = threshold
        self._label_set = label_set
        self._torch = torch
        self._batch_size = max(1, int(batch_size))
        self._use_fp16 = bool(use_fp16)

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            cache_dir=str(cache_dir),
        )
        model = model.to(device).eval()
        if self._use_fp16:
            model = model.half()
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

        self._model = model
        self._preprocess = preprocess

        # Text features are static across the whole run. Encode once,
        # normalize, and cache as fp32 numpy so each image's similarity
        # computation stays on CPU (tiny matmul; crossing the PCIe
        # boundary every frame was the previous waste).
        self._groups: dict[str, tuple[list[str], np.ndarray]] = {}
        with torch.inference_mode():
            for kind, labels in (
                ("object", label_set.objects),
                ("scene", label_set.scenes),
                ("attr", label_set.attrs),
            ):
                if not labels:
                    continue
                prompts = [f"a photo of {lab}" for lab in labels]
                toks = tokenizer(prompts).to(device)
                feats = model.encode_text(toks)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats_np = feats.float().cpu().numpy().astype(np.float32)
                self._groups[kind] = (list(labels), feats_np)

    def tag(self, image_path: Path) -> list[ElementDetection]:
        return self.tag_batch([image_path])[0]

    def tag_batch(self, image_paths: list[Path]) -> list[list[ElementDetection]]:
        """
        Run CLIP on a batch of images. Returns one detection list per input
        path, in input order. Images that fail to decode yield an empty list
        in their slot (the caller's per-image bookkeeping stays aligned).

        If `len(image_paths)` exceeds `batch_size`, the batch is split into
        `batch_size`-sized chunks internally.
        """
        if not image_paths:
            return []
        torch = self._torch

        tensors: list[object | None] = []
        for p in image_paths:
            img = load_rgb_pil(p)
            tensors.append(None if img is None else self._preprocess(img))

        results: list[list[ElementDetection]] = [[] for _ in image_paths]
        valid_idx = [i for i, t in enumerate(tensors) if t is not None]
        if not valid_idx:
            return results

        for start in range(0, len(valid_idx), self._batch_size):
            chunk = valid_idx[start : start + self._batch_size]
            batch = torch.stack([tensors[i] for i in chunk]).to(self._device)
            if self._use_fp16:
                batch = batch.half()
            with torch.inference_mode():
                feats = self._model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.float().cpu().numpy().astype(np.float32)

            for local_i, global_i in enumerate(chunk):
                feat_np = feats_np[local_i]
                detections: list[ElementDetection] = []
                for kind, (labels, text_np) in self._groups.items():
                    sims = text_np @ feat_np
                    scaled = sims * 100.0
                    exp = np.exp(scaled - scaled.max())
                    probs = exp / exp.sum()
                    for lab, p in zip(labels, probs):
                        if float(p) >= self._threshold:
                            detections.append(
                                ElementDetection(kind=kind, label=lab, score=float(p))
                            )
                results[global_i] = detections

        return results
