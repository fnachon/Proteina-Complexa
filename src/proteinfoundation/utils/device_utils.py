"""Backend/device helpers for CUDA, MPS, and CPU execution."""

from __future__ import annotations

import os
import platform

import torch


def _normalize_accelerator_name(value: str | None) -> str:
    requested = (value or "auto").strip().lower()
    aliases = {
        "metal": "mps",
        "apple": "mps",
        "apple_silicon": "mps",
        "nvidia": "gpu",
    }
    return aliases.get(requested, requested)


def _requested_accelerator(preferred: str | None = "auto") -> str:
    """Resolve accelerator request with env override support.

    Environment override:
        COMPLEXA_ACCELERATOR=auto|cpu|gpu|cuda|mps
    """
    env_requested = os.getenv("COMPLEXA_ACCELERATOR")
    return _normalize_accelerator_name(env_requested if env_requested else preferred)


def is_mps_built() -> bool:
    """Return True when this torch build includes MPS support."""
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())


def is_mps_available() -> bool:
    """Return True when MPS backend is available."""
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def is_cuda_available() -> bool:
    """Return True when CUDA backend is available."""
    return torch.cuda.is_available()


def get_mps_unavailable_reason() -> str | None:
    """Return a human-readable reason when MPS is unavailable."""
    if not getattr(torch.backends, "mps", None):
        return "torch.backends.mps is not present in this PyTorch build"
    if not torch.backends.mps.is_built():
        return "PyTorch was not built with MPS support"
    if torch.backends.mps.is_available():
        return None

    # Probe for a concrete runtime message from PyTorch.
    try:
        _ = torch.ones(1, device="mps")
    except Exception as e:
        msg = str(e).strip()
        if msg:
            return msg
    return "torch.backends.mps.is_available() returned False"


def format_backend_diagnostics() -> str:
    """Compact backend diagnostic string for logs."""
    mac_ver = platform.mac_ver()[0] or "unknown"
    machine = platform.machine() or "unknown"
    mps_reason = get_mps_unavailable_reason()
    reason_part = f", mps_reason='{mps_reason}'" if mps_reason else ""
    return (
        f"torch={torch.__version__}, machine={machine}, macOS={mac_ver}, "
        f"cuda_available={is_cuda_available()}, mps_built={is_mps_built()}, "
        f"mps_available={is_mps_available()}{reason_part}"
    )


def get_best_torch_device(preferred: str | None = "auto") -> torch.device:
    """Select torch device honoring COMPLEXA_ACCELERATOR and host availability."""
    requested = _requested_accelerator(preferred)

    if requested in {"gpu", "cuda"}:
        if is_cuda_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "mps":
        return torch.device("mps") if is_mps_available() else torch.device("cpu")

    if requested == "cpu":
        return torch.device("cpu")

    # auto: CUDA -> MPS -> CPU
    if is_cuda_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lightning_accelerator(preferred: str | None = "auto") -> str:
    """Map a preferred accelerator to a valid Lightning accelerator on this host."""
    requested = _requested_accelerator(preferred)
    if requested in {"gpu", "cuda"}:
        if is_cuda_available():
            return "gpu"
        if is_mps_available():
            return "mps"
        return "cpu"
    if requested == "mps":
        return "mps" if is_mps_available() else "cpu"
    if requested == "cpu":
        return "cpu"

    # auto
    if is_cuda_available():
        return "gpu"
    if is_mps_available():
        return "mps"
    return "cpu"


def clear_backend_cache(device: torch.device | str | None = None) -> None:
    """Clear memory cache for CUDA/MPS backends."""
    dev = torch.device(device) if device is not None else get_best_torch_device()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        return
    if dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
