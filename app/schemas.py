###############################################################
#
#   Defines pydantic schemas for in-/output.
#
###############################################################

from uuid import UUID, uuid4
from typing import List, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field

from chemnetwork.data.point_clouds import TargetDistribution
from app.config import settings


class AvailableResponse(BaseModel):
    """API response listing the served target distributions incl. labels and the default.
    """
    targets: List[Any]
    default: Dict[str, Any]


class GenerateRequest(BaseModel):
    """API request body for MLP input.
    """
    num_samples: int            = Field(default=500, ge=1, le=5_000)
    integration_steps: int      = Field(default=100, ge=2, le=1_000)
    target: TargetDistribution  = Field(default=settings.default_target)
    model_type: str             = Field(default=settings.default_modeltype)     # TODO: Make this field some enum class too!      
    device: str                 = Field(default="cpu")


class GenerateRequestDB(GenerateRequest):
    """Model to save the request model to database.
    """
    id: UUID = Field(default_factory=uuid4, alias="_id")    # NOTE: Pydantic serializer does not like UUID -> str
    time: datetime


class GenerateResponse(BaseModel):
    """API response body for MLP output.
    """
    num_samples: int
    target: TargetDistribution
    model_type: str  # TODO: Make this field Enum
    device_requested: str
    device_used: str
    source_points: List[List[float]]
    generated_points: List[List[List[float]]]  # Shape (integration_steps, num_samples, 2) or (1, num_samples, 2).
    target_points: List[List[float]]


class GenerateResponseDB(GenerateResponse):     # TODO: Really save all points of the response? How much space does it occupy?
    """Model to save response model to database.
    """
    id: UUID = Field(default_factory=uuid4, alias="_id")    # NOTE: Pydantic serializer does not like UUID -> str
    time: datetime