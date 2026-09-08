###############################################################
#
#   This module defines a registry for managing multiple pre-trained models
#
###############################################################
from pathlib import Path
from collections.abc import Iterable
from chemnetwork.sample import load_model, LoadedTuple
from chemnetwork.data.point_clouds import TargetDistribution

class ModelRegistry:
    """A registry for managing multiple pre-trained models, each associated with a specific target distribution.
    """

    def __init__(self, loaded: dict) -> None:
        self._loaded = loaded

    def __len__(self) -> int:
        return len(self._loaded)
    
    def __contains__(self, target: str) -> bool:
        return target in self._loaded

    @classmethod
    def from_checkpoints(cls, paths: Iterable[Path]) -> "ModelRegistry":
        """Associates model checkpoints and config to a specific name for the model
            from a list of checkpoint paths.
            Note: This makes it possible to read the keys from the checkpoint and not hand-write everything!

            Args:
                paths (Iterable[Path]): List of paths to the .pt checkpoint files.

            Raises:
                ValueError: If a checkpoint claims a target name that is not a known
                    TargetDistribution, or if two checkpoints claim the same target.

            Returns:
                ModelRegistry: An instance of ModelRegistry containing the loaded models and their configurations.
        """
        loaded = {}
        for path in paths:
            entry = load_model(checkpoint_path=path)
            name = entry.config["data"]["target_distribution"]

            try:
                target = TargetDistribution(name)
            except ValueError as error:
                raise ValueError(
                    f"Checkpoint '{path}' claims unknown target '{name}'. "
                    f"Known targets: {[t.value for t in TargetDistribution]}."
                ) from error

            if target in loaded:
                raise ValueError(f"Two checkpoints claim target '{target.value}'.")
            loaded[target.value] = entry  # Assign (model, cfg) tuple to the target name to get a unique coupling => {model_target_name: (model=loaded_model, config=dict_cfg), ...}
        return cls(loaded)

    @property
    def available(self) -> list[str]:
        """Lists the available loaded models.

            Returns:
                list[str]: A list of target distribution names for which models are available in the registry.
        """
        return sorted(self._loaded)

    def get(self, target: str) -> LoadedTuple:
        """Get the tuple (model, cfg) for a defined target distribution.

            Args:
                target (str): Name of the target distribution, e.g., "moons" or "checkerboard".

            Returns:
                LoadedTuple: A tuple containing the loaded model and its configuration.
        """
        return self._loaded[target]

    def clear(self) -> None:
        """Clean up the models and release the resources
        """
        self._loaded.clear()