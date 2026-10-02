"""
Explicit device selection shared by src/train.py and the sweep script.

"auto" keeps the original selection order (CUDA, then MPS, then CPU). An
explicitly requested device that is not available raises instead of silently
falling back, so a run labelled "cuda" can never have executed on something
else.
"""

from __future__ import annotations

import platform
import sys

import torch

DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")


def resolve_device(requested: str = "auto") -> str:
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"unknown device {requested!r}; expected one of {DEVICE_CHOICES}")
    if requested == "auto":
        return (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device 'cuda' was requested but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}, built with CUDA {torch.version.cuda}). "
            "Use --device auto, mps or cpu on this machine."
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "device 'mps' was requested but torch.backends.mps.is_available() is False "
            f"(torch {torch.__version__}, MPS built: {torch.backends.mps.is_built()}). "
            "Use --device auto, cuda or cpu on this machine."
        )
    return requested


def device_metadata(requested: str, resolved: str) -> dict:
    """Hardware/software provenance stored alongside every result."""
    info = {
        "requested": requested,
        "resolved": resolved,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_name": None,
    }
    if resolved == "cuda":
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return info
