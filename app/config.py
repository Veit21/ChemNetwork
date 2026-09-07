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

    # General
    model_config = SettingsConfigDict(env_prefix="CHEMNET_")
    name: str = "ChemNetwork"

    # Model settings
    checkpoint_paths: list[Path] = [
        Path("serving_checkpoint/MultiLayerPerceptron_moons_weights_step_50000.pt"),
        Path("serving_checkpoint/MultiLayerPerceptron_checkerboard_weights_step_50000.pt"),
    ]
    default_target: TargetDistribution = TargetDistribution.MOONS
    integration_steps: int = 100                            # This is the actual number of steps used by the ODE solver to integrate the learned vector field.
    max_trajectory_steps: int = Field(default=50, ge=2)     # This is important for the front-end visualization, as it limits the number of frames to animate. This does not influence the actual ODE integration!

    # DB settings
    db_name: str = "io_database"
    db_host_name : str = "mongodb"      # NOTE: References the mongodb service (i.e. the database container)in docker-compose.yaml
    db_port: int = 27017
    connection_timeout_ms: int = 3000   # How many ms to try to connect to database upon startup
    request_collection_name: str = "request_collection"
    response_collection_name: str = "response_collection"

    @property
    def db_uri(self):
        return f"mongodb://{self.db_host_name}:{self.db_port}/"


settings = Settings()