###############################################################
#
#   Defines pydantic schemas for in-/output.
#
###############################################################

from pydantic import BaseModel, Field
from typing import List, Union
from chemnetwork.data.point_clouds import TargetDistribution


class GenerateRequest(BaseModel):
    """API request template.
    """
    num_samples: int = Field(default=500, ge=1, le=5_000)
    integration_steps: int = Field(default=100, ge=2, le=1_000)
    return_trajectory: bool = Field(default=False)
    target: TargetDistribution = TargetDistribution.MOONS

class GenerateResponse(BaseModel):
    """API response template.
    """
    num_samples: int
    target: TargetDistribution  # TODO: Do not forget in app.js
    source_points: List[List[float]]
    generated_points: List[List[List[float]]]  # If return_trajectory is True, the shape will be (integration_steps, num_samples, 2), otherwise (num_samples, 2).
    target_points: List[List[float]]
