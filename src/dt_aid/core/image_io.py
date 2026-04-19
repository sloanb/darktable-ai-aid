from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

RAW_EXTS = {
    ".dng", ".raf", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
    ".orf", ".rw2", ".pef", ".raw", ".rwl", ".srw", ".3fr", ".erf", ".mef",
    ".mos", ".mrw", ".x3f",
}


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTS


def load_bgr(path: Path) -> np.ndarray | None:
    """
    Load any supported image as a BGR uint8 numpy array. For RAW formats
    we prefer the embedded JPEG preview (fast, already demosaiced by the
    camera) and fall back to a half-size LibRaw demosaic if no preview is
    available. Returns None if the file can't be decoded.
    """
    if is_raw(path):
        return _load_raw(path)
    img = cv2.imread(str(path))
    if img is None:
        log.warning("cv2.imread returned None for %s", path)
    return img


def _load_raw(path: Path) -> np.ndarray | None:
    try:
        import rawpy
    except ImportError:
        log.error("rawpy is not installed; cannot decode RAW file %s", path)
        return None

    try:
        with rawpy.imread(str(path)) as raw:
            try:
                thumb = raw.extract_thumb()
            except (
                rawpy.LibRawNoThumbnailError,
                rawpy.LibRawUnsupportedThumbnailError,
            ):
                rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=False)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if thumb.format == rawpy.ThumbFormat.JPEG:
                arr = np.frombuffer(thumb.data, dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    return decoded
                log.warning("failed to decode embedded JPEG thumb for %s", path)
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                return cv2.cvtColor(thumb.data, cv2.COLOR_RGB2BGR)

            rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=False)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        log.warning("failed to decode RAW %s: %s", path, e)
        return None


def load_rgb_pil(path: Path):
    """Return a PIL.Image.Image (RGB), using the same RAW handling as load_bgr."""
    from PIL import Image

    bgr = load_bgr(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
