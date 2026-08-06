###############################################################
#
#   Defines pydantic schemas for in-/output.
#
###############################################################

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    num_samples: int = Field(default=500, ge=1, le=5_000)
    integration_steps: int = Field(default=100, ge=1, le=1_000)

class GenerateResponse(BaseModel):
    num_samples: int
    points: list[list[float]]
