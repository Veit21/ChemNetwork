###############################################################
#
#   Defines pydantic schemas for in-/output.
#
###############################################################

from pydantic import BaseModel, Field
from typing import List, Union


class GenerateRequest(BaseModel):
    """API request template.
    """
    num_samples: int = Field(default=500, ge=1, le=5_000)
    integration_steps: int = Field(default=100, ge=2, le=1_000)
    return_trajectory: bool = Field(default=False)

class GenerateResponse(BaseModel):
    """API response template.
    """
    num_samples: int
    source_points: List[List[float]]    # TODO: Also add the ground truth points to the response, so that they can be plotted in the frontend.
    generated_points: List[List[List[float]]]  # If return_trajectory is True, the shape will be (integration_steps, num_samples, 2), otherwise (num_samples, 2).
