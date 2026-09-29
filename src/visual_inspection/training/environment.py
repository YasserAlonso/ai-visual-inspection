"""Lazy hardware/framework inspection so Phase 0 installs stay lightweight."""

import platform


def environment_metadata() -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Training dependencies are missing; install the training extra") from exc
    cuda_version = torch.version.cuda
    available = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if available else None
    gpu_memory = {}
    if available:
        free, total = torch.cuda.mem_get_info(0)
        gpu_memory = {
            "gpu_memory_free_mib_at_start": round(free / (1024**2), 1),
            "gpu_memory_total_mib": round(total / (1024**2), 1),
            "gpu_compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        }
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": available,
        "cuda_version": cuda_version,
        "gpu_name": gpu,
        **gpu_memory,
    }


def select_device(requested: str | int, metadata: dict | None = None) -> str:
    metadata = metadata or environment_metadata()
    if requested == "auto":
        return "auto" if metadata["cuda_available"] else "cpu"
    if requested == "cpu":
        return "cpu"
    if isinstance(requested, int):
        if not metadata["cuda_available"]:
            raise ValueError("CUDA device requested but CUDA is unavailable")
        return str(requested)
    if isinstance(requested, str) and requested.isdigit():
        if not metadata["cuda_available"]:
            raise ValueError("CUDA device requested but CUDA is unavailable")
        return requested
    raise ValueError("device must be auto, cpu, or a CUDA device index")
