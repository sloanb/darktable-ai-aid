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
    (objects, scenes, attrs) the tagger returns labels whose cosine
    similarity against the image embedding exceeds `threshold`, converted
    to a softmax-normalized probability within the group.
    """

    def __init__(
        self,
        *,
        label_set: LabelSet,
        cache_dir: Path,
        threshold: float = 0.25,
        device: str | None = None,
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

        self._device = device
        self._threshold = threshold
        self._label_set = label_set
        self._torch = torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            cache_dir=str(cache_dir),
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

        self._model = model
        self._preprocess = preprocess

        self._groups: dict[str, tuple[list[str], "torch.Tensor"]] = {}
        with torch.no_grad():
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
                self._groups[kind] = (labels, feats)

    def tag(self, image_path: Path) -> list[ElementDetection]:
        torch = self._torch
        img = load_rgb_pil(image_path)
        if img is None:
            return []
        x = self._preprocess(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feat_np = feat.cpu().numpy().astype(np.float32)[0]

        out: list[ElementDetection] = []
        for kind, (labels, text_feats) in self._groups.items():
            text_np = text_feats.cpu().numpy().astype(np.float32)
            sims = text_np @ feat_np  # [num_labels]
            # softmax over the group, scaled (CLIP convention uses ~100)
            scaled = sims * 100.0
            exp = np.exp(scaled - scaled.max())
            probs = exp / exp.sum()
            for lab, p in zip(labels, probs):
                if float(p) >= self._threshold:
                    out.append(ElementDetection(kind=kind, label=lab, score=float(p)))
        return out
