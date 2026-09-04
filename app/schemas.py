###############################################################
#
#   Defines pydantic schemas for in-/output.
#
###############################################################

from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

from chemnetwork.data.point_clouds import TargetDistribution
from app.config import settings


TARGET_LABELS: dict[TargetDistribution, str] = {
    TargetDistribution.MOONS: "Two moons",
    TargetDistribution.CHECKERBOARD: "Checkerboard",
}


class TargetInfo(BaseModel):    # TODO: Maybe remove this again? Is this so necessary? Another solution?
    """One target distribution that has a served model.
    """
    id: TargetDistribution
    label: str

    @classmethod
    def from_target(cls, target: TargetDistribution) -> "TargetInfo":
        """Describes a target distribution for the frontend.

            Args:
                target (TargetDistribution): The target distribution to describe.

            Returns:
                TargetInfo: The machine key plus a human-readable label. Targets with no
                entry in TARGET_LABELS fall back to a prettified version of their value.
        """
        return cls(
            id=target,
            label=TARGET_LABELS.get(target, target.value.replace("_", " ").capitalize()),
        )


class AvailableResponse(BaseModel):
    """API response listing every target distribution the registry can serve.
    """
    targets: List[TargetInfo]
    default: TargetDistribution

class GenerateRequest(BaseModel):
    """API request body for MLP input.
    """
    num_samples: int            = Field(default=500, ge=1, le=5_000)
    integration_steps: int      = Field(default=100, ge=2, le=1_000)
    return_trajectory: bool     = Field(default=True)
    target: TargetDistribution  = Field(default=settings.default_target)
    device: str                 = Field(default="cpu")

class GenerateRequestDB(GenerateRequest):
    """Model to save the request model to database.
    """
    time: datetime

class GenerateResponse(BaseModel):
    """API response body for MLP output.
    """
    num_samples: int
    target: TargetDistribution
    device_requested: str
    device_used: str
    source_points: List[List[float]]
    generated_points: List[List[List[float]]]  # If return_trajectory is True, the shape will be (integration_steps, num_samples, 2), otherwise (1, num_samples, 2).
    target_points: List[List[float]]

class GenerateResponseDB(GenerateResponse):     # TODO: Really save all points of the response? How much space does it occupy?
    """Model to save response model to database.
    """
    time: datetime