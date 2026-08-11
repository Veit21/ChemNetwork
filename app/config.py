###############################################################
#
#   Defines the config settings for loading a model.
#
###############################################################

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHEMNET_")

    checkpoint_path: Path = Path("checkpoints/MultiLayerPerceptron_weights_step_50000.pt")
    integration_steps: int = 100                            # This is the actual number of steps used by the ODE solver to integrate the learned vector field.
    max_trajectory_steps: int = Field(default=50, ge=1)     # This is important for the front-end visualization, as it limits the number of frames to animate. This does not influence the actual ODE integration!

settings = Settings()