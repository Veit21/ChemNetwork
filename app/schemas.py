###############################################################
#
#   Defines pydantic schemas for in-/output.
#
###############################################################

import uuid

from typing import List, Any
from datetime import datetime
from pydantic import BaseModel, Field

from chemnetwork.data.point_clouds import TargetDistribution
from app.config import settings


class AvailableResponse(BaseModel):
    """API response listing the served target distributions incl. labels and the default.
    """
    targets: List[Any]
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
    id: str = Field(default_factory=uuid.uuid4, alias="_id")    # NOTE: Pydantic serializer does not like UUID -> str
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
    id: str = Field(default_factory=uuid.uuid4, alias="_id")    # NOTE: Pydantic serializer does not like UUID -> str
    time: datetime