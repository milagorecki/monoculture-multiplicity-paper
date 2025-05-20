import torch
import os
from torch import Tensor


def process_device(device: str) -> str:
    """Set and return the default device to cpu or gpu (cuda, mps).

    Args:
        device: target torch device
    Returns:
        device: processed string, e.g., "cuda" is mapped to "cuda:0".
    """

    if device == "cpu":
        return "cpu"
    else:
        # If user just passes 'gpu', search for CUDA or MPS.
        if device == "gpu":
            # check whether either pytorch cuda or mps is available
            if torch.cuda.is_available():
                current_gpu_index = torch.cuda.current_device()
                device = f"cuda:{current_gpu_index}"
                check_device(device)
                torch.cuda.set_device(device)
            elif torch.backends.mps.is_available():
                device = "mps:0"
                # MPS support is not implemented for a number of operations.
                # use CPU as fallback.
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                # MPS framework does not support double precision.
                torch.set_default_dtype(torch.float32)
                check_device(device)
            else:
                raise RuntimeError(
                    "Neither CUDA nor MPS is available. "
                    "Please make sure to install a version of PyTorch that supports "
                    "CUDA or MPS."
                )
        # Else, check whether the custom device is valid.
        else:
            check_device(device)

        return device


def gpu_available() -> bool:
    """Check whether GPU is available."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def check_device(device: str) -> None:
    """Check whether the device is valid.

    Args:
        device: target torch device
    """
    try:
        torch.randn(1, device=device)
    except (RuntimeError, AssertionError) as exc:
        raise RuntimeError(
            f"""Could not instantiate torch.randn(1, device={device}). Make sure
             the device is set up properly and that you are passing the
             corresponding device string. It should be something like 'cuda',
             'cuda:0', or 'mps'. Error message: {exc}."""
        ) from exc


def atleast_2d(t: Tensor) -> Tensor:
    return t if t.ndim >= 2 else t.reshape(1, -1)


def vectorize_1d(t: Tensor) -> Tensor:
    return t if (t.ndim == 2) and (t.shape[1] == 1) else t.reshape(-1, 1)


def assert_all_finite(quantity: Tensor, description: str = "tensor") -> None:
    """Raise if tensor quantity contains any NaN or Inf element."""

    msg = f"NaN/Inf present in {description}."
    assert torch.isfinite(quantity).all(), msg
