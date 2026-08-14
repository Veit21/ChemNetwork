import pytest
from pathlib import Path

from app.model_registry import ModelRegistry
from app.config import settings

def test_model_registry_available_targets() -> None:
    registry = ModelRegistry.from_checkpoints(paths=settings.checkpoint_paths)
    available_targets = registry.available
    assert "moons" in available_targets
    assert "checkerboard" in available_targets
    assert len(available_targets) == 2