import pytest

from app.model_registry import ModelRegistry
from app.config import settings

# TODO: This is implicitly tested in "test_samples.py" anyways, right? Remove?
@pytest.mark.skip(reason="For later.")
def test_model_registry_available_targets() -> None:
    registry = ModelRegistry.from_checkpoints(paths=settings.checkpoint_paths)  # TODO: Class does not find the relative paths, how to fix?
    available_targets = registry.available
    assert "moons" in available_targets
    assert "checkerboard" in available_targets
    assert len(available_targets) == 2