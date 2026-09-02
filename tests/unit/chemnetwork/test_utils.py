import pytest
import torch

from chemnetwork.utils import resolve_device


@pytest.mark.parametrize("requested", ["cpu", "CPU", " cpu "])
def test_resolve_device_cpu(requested: str) -> None:
    """The CPU is always available and should be resolved regardless of casing/whitespace.
    """
    assert resolve_device(requested) == torch.device("cpu")


def test_resolve_device_auto_picks_available_accelerator() -> None:
    """"auto" should pick CUDA/MPS when present and the CPU otherwise.
    """
    resolved = resolve_device("auto")

    if torch.cuda.is_available():
        assert resolved.type == "cuda"
    elif torch.backends.mps.is_available():
        assert resolved.type == "mps"
    else:
        assert resolved.type == "cpu"


def test_resolve_device_falls_back_to_cpu_without_cuda(monkeypatch) -> None:
    """Requesting CUDA on a machine without a GPU must not crash the run.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("cuda") == torch.device("cpu")


def test_resolve_device_rejects_garbage() -> None:
    """An uninterpretable device string should raise a ValueError instead of failing later.
    """
    with pytest.raises(ValueError):
        resolve_device("gpu")
