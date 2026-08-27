###############################################################
#
#   Defines the config settings for loading a model.
#
###############################################################

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from chemnetwork.data.point_clouds import TargetDistribution


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHEMNET_")

    checkpoint_paths: list[Path] = [
        Path("serving_checkpoint/MultiLayerPerceptron_moons_weights_step_50000.pt"),
        Path("serving_checkpoint/MultiLayerPerceptron_checkerboard_weights_step_50000.pt"),
    ]
    default_target: TargetDistribution = TargetDistribution.MOONS
    integration_steps: int = 100                            # This is the actual number of steps used by the ODE solver to integrate the learned vector field.
    max_trajectory_steps: int = Field(default=50, ge=2)     # This is important for the front-end visualization, as it limits the number of frames to animate. This does not influence the actual ODE integration!

settings = Settings()