from __future__ import annotations

import ctypes
import glob
import logging
import os
import sys

log = logging.getLogger(__name__)

_PRELOADED = False

_NVIDIA_LIBS = [
    ("nvidia/cuda_runtime/lib", "libcudart.so*"),
    ("nvidia/cuda_nvrtc/lib", "libnvrtc.so*"),
    ("nvidia/nvjitlink/lib", "libnvJitLink.so*"),
    ("nvidia/cublas/lib", "libcublas.so*"),
    ("nvidia/cublas/lib", "libcublasLt.so*"),
    ("nvidia/cudnn/lib", "libcudnn.so*"),
    ("nvidia/cudnn/lib", "libcudnn_*.so*"),
    ("nvidia/cufft/lib", "libcufft.so*"),
    ("nvidia/curand/lib", "libcurand.so*"),
    ("nvidia/cusolver/lib", "libcusolver.so*"),
    ("nvidia/cusparse/lib", "libcusparse.so*"),
    ("nvidia/nccl/lib", "libnccl.so*"),
    ("nvidia/nvtx/lib", "libnvToolsExt.so*"),
]


def preload_cuda_libs() -> bool:
    """
    Dlopen the bundled NVIDIA CUDA 12 shared libraries from the venv's
    site-packages into the global symbol space so onnxruntime's CUDA
    provider can find them without LD_LIBRARY_PATH. Idempotent; safe to
    call every scan start. Returns True if at least one library was
    preloaded, False if the packages aren't installed.
    """
    global _PRELOADED
    if _PRELOADED:
        return True
    site_pkgs: list[str] = []
    for p in sys.path:
        if p.endswith("site-packages") and os.path.isdir(os.path.join(p, "nvidia")):
            site_pkgs.append(p)
    if not site_pkgs:
        return False
    loaded_any = False
    for site in site_pkgs:
        for subdir, pattern in _NVIDIA_LIBS:
            for so in sorted(glob.glob(os.path.join(site, subdir, pattern))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    loaded_any = True
                except OSError as e:
                    log.debug("preload skipped %s: %s", so, e)
    _PRELOADED = loaded_any
    return loaded_any


def resolve_onnx_providers(device: str) -> list[str]:
    """
    Translate a user-facing device preference into an ONNX Runtime
    provider list. Preloads bundled NVIDIA CUDA 12 libs from the venv
    before probing, so Blackwell/CUDA systems work without manually
    setting LD_LIBRARY_PATH. Preferences:
      - "cuda": require CUDA; raise if unavailable.
      - "cpu":  CPU only.
      - "auto": CUDA if importable, else CPU. Logs the decision.
    """
    device = device.lower()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    preload_cuda_libs()
    if device == "cuda":
        if not _cuda_available():
            raise RuntimeError(
                "device=cuda requested but CUDAExecutionProvider is not available. "
                "Install onnxruntime-gpu: `pip install -e '.[gpu]'` and ensure NVIDIA drivers are current."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # auto
    if _cuda_available():
        log.info("device=auto: using CUDAExecutionProvider")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    log.info("device=auto: CUDA unavailable, falling back to CPU")
    return ["CPUExecutionProvider"]


def resolve_torch_device(device: str) -> str:
    """User-facing device preference → torch device string."""
    device = device.lower()
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not _torch_cuda_available():
            raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if _torch_cuda_available() else "cpu"


def _cuda_available() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())
