from __future__ import annotations

import sys
from pathlib import Path

from platformdirs import user_data_dir
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

FACES_MODEL_NAME = "buffalo_l"
FACES_MODEL_VERSION = "buffalo_l-v1"
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
ELEMENTS_MODEL_VERSION = f"openclip-{CLIP_MODEL_NAME}-{CLIP_PRETRAINED}-v1"


def default_darktable_library() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library/application support/darktable/library.db"
    return Path.home() / ".config/darktable/library.db"


def default_data_dir() -> Path:
    return Path(user_data_dir("dt-aid", appauthor=False))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DT_AID_", extra="ignore")

    darktable_library: Path = Field(default_factory=default_darktable_library)
    data_dir: Path = Field(default_factory=default_data_dir)
    known_faces_dir: Path | None = None

    face_match_threshold: float = 0.5
    face_det_size: int = 640
    face_det_score_threshold: float = 0.5

    elements_threshold: float = 0.25
    elements_labels_file: Path | None = None
    # Images per CLIP forward pass. Larger = better GPU utilization but
    # more VRAM; 16 is comfortable at fp16 on an 8 GB GPU.
    elements_batch_size: int = 16

    # "auto" picks CUDA when onnxruntime-gpu is importable, else CPU.
    device: str = "auto"  # auto | cpu | cuda

    @property
    def state_db(self) -> Path:
        return self.data_dir / "state.db"

    @property
    def faces_dir(self) -> Path:
        return self.data_dir / "faces"

    @property
    def face_embeddings_npy(self) -> Path:
        return self.faces_dir / "embeddings.npy"

    @property
    def face_embeddings_meta(self) -> Path:
        return self.faces_dir / "embeddings.meta.parquet"

    @property
    def face_references_dir(self) -> Path:
        return self.faces_dir / "references"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.faces_dir,
            self.face_references_dir,
            self.models_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def load_settings(**overrides) -> Settings:
    return Settings(**overrides)
