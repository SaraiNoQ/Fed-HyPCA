"""Centralized GPU selection for Fed-Hy.

Manages CUDA device visibility and provides consistent device references
for both PyTorch and HuggingFace model loading.

Usage in entry-point scripts (MUST come before any torch import):

    from src.utils.gpu import configure_gpu_from_args
    configure_gpu_from_args(default=1)   # parses --gpu_id / FED_HY_GPU_ID

    import torch                          # now safe -- only sees 1 GPU
    from src.utils.gpu import get_device, get_device_map
"""

import logging
import os

logger = logging.getLogger(__name__)

_configured: bool = False
_gpu_id: int | None = None


def configure_gpu(gpu_id: int = None) -> None:
    """Set CUDA_VISIBLE_DEVICES for the current process.

    Must be called BEFORE any torch import.

    Args:
        gpu_id: Physical GPU index. Resolution order if None:
                1. FED_HY_GPU_ID environment variable
                2. CUDA_VISIBLE_DEVICES if already set externally
                3. Default to 0
    """
    global _configured, _gpu_id

    if _configured:
        raise RuntimeError(
            f"configure_gpu() already called with gpu_id={_gpu_id}. "
            "GPU configuration is process-global and cannot be changed."
        )

    if gpu_id is not None:
        resolved_id = gpu_id
    elif "FED_HY_GPU_ID" in os.environ:
        resolved_id = int(os.environ["FED_HY_GPU_ID"])
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        _configured = True
        _gpu_id = None
        logger.info(
            "CUDA_VISIBLE_DEVICES already set externally to '%s', not overriding.",
            os.environ["CUDA_VISIBLE_DEVICES"],
        )
        return
    else:
        resolved_id = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(resolved_id)
    _configured = True
    _gpu_id = resolved_id
    logger.info(
        "GPU configured: CUDA_VISIBLE_DEVICES=%s (physical GPU %d appears as cuda:0)",
        resolved_id, resolved_id,
    )


def configure_gpu_from_args(default: int = 0, arg_name: str = "--gpu_id") -> None:
    """Parse gpu_id from sys.argv or env var, then call configure_gpu().

    Resolution order:
    1. arg_name in sys.argv (e.g., --gpu_id 1)
    2. FED_HY_GPU_ID environment variable
    3. default parameter
    """
    import sys

    gpu_id = default
    found_in_argv = False

    for i, arg in enumerate(sys.argv):
        if arg == arg_name and i + 1 < len(sys.argv):
            gpu_id = int(sys.argv[i + 1])
            found_in_argv = True
            break

    if not found_in_argv and "FED_HY_GPU_ID" in os.environ:
        gpu_id = int(os.environ["FED_HY_GPU_ID"])

    configure_gpu(gpu_id=gpu_id)


def get_device() -> "torch.device":
    """Return the torch device for computation.

    Returns cuda:0 (the single visible GPU after configure_gpu), or cpu.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_device_map() -> dict:
    """Return device_map for HuggingFace from_pretrained().

    Returns {"": "cuda:0"} to load the entire model onto the single
    visible GPU. Replaces device_map="auto" which would auto-distribute
    across multiple GPUs.
    """
    import torch

    if torch.cuda.is_available():
        return {"": "cuda:0"}
    return {"": "cpu"}
