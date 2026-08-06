###############################################################
#
#   Defines the config settings for loading a model.
#
###############################################################

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHEMNET_")

    checkpoint_path: Path = Path("checkpoints/MultiLayerPerceptron_weights_step_50000.pt")
    integration_steps: int = 100
    max_samples: int = 5_000

settings = Settings()