from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass(frozen=True)
class LabelSet:
    objects: list[str]
    scenes: list[str]
    attrs: list[str]


DEFAULT_OBJECTS = [
    "person", "dog", "cat", "bird", "horse", "car", "bicycle", "motorcycle",
    "boat", "airplane", "train", "flower", "tree", "food", "computer", "book",
    "chair", "couch", "bed", "dining table",
]

DEFAULT_SCENES = [
    "beach", "mountain", "forest", "city street", "indoors", "kitchen",
    "bedroom", "office", "restaurant", "park", "desert", "snow", "underwater",
    "sky", "sunset", "sunrise", "night scene",
]

DEFAULT_ATTRS = [
    "black and white", "high contrast", "low light", "portrait", "landscape",
    "close-up", "wide shot", "silhouette", "group photo",
]


def default_label_set() -> LabelSet:
    return LabelSet(
        objects=list(DEFAULT_OBJECTS),
        scenes=list(DEFAULT_SCENES),
        attrs=list(DEFAULT_ATTRS),
    )


def load_label_set(path: Path) -> LabelSet:
    """
    Load labels from a TOML file:
      [labels]
      objects = ["dog", "cat"]
      scenes = ["beach"]
      attrs = ["sunset"]
    Missing keys fall back to defaults.
    """
    data = tomllib.loads(path.read_text())
    labels = data.get("labels", {})
    defaults = default_label_set()
    return LabelSet(
        objects=list(labels.get("objects", defaults.objects)),
        scenes=list(labels.get("scenes", defaults.scenes)),
        attrs=list(labels.get("attrs", defaults.attrs)),
    )
