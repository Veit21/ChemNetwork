from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from datetime import datetime

from chemnetwork.sample import generate_samples
from chemnetwork.utils import resolve_device
from chemnetwork.data.point_clouds import TargetDistribution
from app.schemas import AvailableResponse, GenerateRequest, GenerateResponse, TargetInfo, GenerateRequestDB, GenerateResponseDB
from app.dependencies import get_model_registry, get_model_database
from app.serialization import downsample_trajectory_tensor, typecast_and_round_output
from app.config import settings
from app.model_registry import ModelRegistry
from pymongo.synchronous.database import Database


# Set router parameters
router = APIRouter(
    prefix="/samples",
    tags=["samples"],
)

@router.get("/available", response_model=AvailableResponse)
def available(
    registry: Annotated[ModelRegistry, Depends(get_model_registry)]
) -> AvailableResponse:
    """API endpoint to retrieve the learned target distributions.

        Args:
            registry (Annotated[ModelRegistry, Depends): The model registry that holds the data for all loaded models.

        Returns:
            AvailableResponse: A Pydantic model containing the loaded targets and the config default target.
    """
    return AvailableResponse(
        targets=[TargetInfo.from_target(TargetDistribution(target)) for target in registry.available],
        default=settings.default_target,
    )

@router.post("/generate", response_model=GenerateResponse)
def generate(
    req: GenerateRequest,
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    database: Annotated[Database, Depends(get_model_database)],
) -> GenerateResponse:
    """_summary_

        Args:
            req (GenerateRequest): _description_
            registry (Annotated[ModelRegistry, Depends): _description_
            database (Annotated[Database, Depends): _description_

        Raises:
            HTTPException: _description_

        Returns:
            GenerateResponse: _description_
    """
    try:

        # Retrieve the model that generates the desired target data
        entry = registry.get(req.target)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model served for '{req.target.value}'. Available: {registry.available}."
        ) 

    # TODO: 1. So far, falling back to CPU silently!
    # TODO: 2. Enable CUDA and MPS support in Docker container!
    # Resolve the computation device
    device = resolve_device(req.device)

    # Generate samples
    samples = generate_samples(
        model               = entry.model,
        cfg                 = entry.config,
        num_samples         = req.num_samples,
        integration_steps   = req.integration_steps,
        return_trajectory   = req.return_trajectory,
        device              = device
    )
    
    # Subsample the trajectory
    # TODO: Subsampling still breaks if return_trajectory=False. Maybe always make the network return the trajectory.
    predicted_data_subsampled = downsample_trajectory_tensor(
        trajectory  = samples.generated,
        max_steps   = settings.max_trajectory_steps
    )

    # Instantiate the response
    response = GenerateResponse(
        num_samples         = req.num_samples,
        source_points       = typecast_and_round_output(samples.source),
        generated_points    = typecast_and_round_output(predicted_data_subsampled),
        target_points       = typecast_and_round_output(samples.target),
        target              = req.target,
        device_requested    = req.device,
        device_used         = device.type,
        )

    # TODO: Make sure the database is even connected! Solve with some kind of boolean handle.
    # Save request and response to database
    db_entry_time       = datetime.now()
    request_db_entry    = GenerateRequestDB(**req.model_dump(), time=db_entry_time)
    response_db_entry   = GenerateResponseDB(**response.model_dump(), time=db_entry_time)
    new_request_entry   = database[settings.request_collection_name].insert_one(jsonable_encoder(request_db_entry))
    # new_response_entry = database[settings.response_collection_name].insert_one(jsonable_encoder(response_db_entry))

    return response