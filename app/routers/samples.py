from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from datetime import datetime

from chemnetwork.sample import generate_samples
from chemnetwork.utils import resolve_device
from chemnetwork.data.point_clouds import TargetDistribution
from app.schemas import AvailableResponse, GenerateRequest, GenerateResponse, GenerateRequestDB, GenerateResponseDB
from app.dependencies import get_model_registry, get_model_database
from app.serialization import downsample_trajectory_tensor, typecast_and_round_output
from app.config import settings
from app.model_registry import ModelRegistry
from pymongo.synchronous.database import Database
from pymongo import errors


# Set router parameters
router = APIRouter(
    prefix="/samples",
    tags=["samples"],
)

@router.get("/available", response_model=AvailableResponse)
def available(
    registry: Annotated[ModelRegistry, Depends(get_model_registry)]
) -> AvailableResponse:
    """API endpoint to list the available targets, models served for each target and the default combination of (target, model).

    Args:
        registry (Annotated[ModelRegistry, Depends): Model registry containing information about the served models.

    Returns:
        AvailableResponse: Response model that structures the list of targets and models.
    """
    return AvailableResponse(
        targets=[{"id": TargetDistribution(target), "label": TargetDistribution.as_label(target), "models": models} for target, models in registry.available_models_for_targets.items()],
        default={"target": settings.default_target, "model": settings.default_modeltype},
    )

@router.post("/generate", response_model=GenerateResponse)
def generate(
    req: GenerateRequest,
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    database: Annotated[Database, Depends(get_model_database)],
) -> GenerateResponse:
    """API endpoint to generate samples via the backbone neural network.

        Args:
            req (GenerateRequest): Pydantic model for the request body.
            registry (Annotated[ModelRegistry, Depends): Model registry containing the loaded models/checkpoints.
            database (Annotated[Database, Depends): Database object to which requests and responses are saved.

        Raises:
            HTTPException: If there is no served model for the desired target distribution.

        Returns:
            GenerateResponse: Pydantic model for the response body.
    """

    # Retrieve the model that generates the desired target data
    try:
        entry = registry.get(target=req.target, model_type=req.model_type)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{req.model_type}' not served for target '{req.target.value}'. Available: {registry.available_models_for_targets}."
        ) 

    # TODO: 1. So far, falling back to CPU silently!
    # Resolve device
    device = resolve_device(req.device)

    # Generate samples
    try:
        samples = generate_samples(
            model               = entry.model,
            cfg                 = entry.config,
            model_type          = req.model_type,
            num_samples         = req.num_samples,
            integration_steps   = req.integration_steps,
            device              = device
        )
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"No solver implemented for requested model '{req.model_type}'."
        )
    
    # Subsample the trajectory
    if req.model_type == "flow":  # TODO: Enum for model type
        samples_generated = downsample_trajectory_tensor(
            trajectory  = samples.generated,
            max_steps   = settings.max_trajectory_steps
        )
    else:
        samples_generated = samples.generated

    # Instantiate the response
    response = GenerateResponse(
        num_samples         = req.num_samples,
        source_points       = typecast_and_round_output(samples.source),
        generated_points    = typecast_and_round_output(samples_generated),
        target_points       = typecast_and_round_output(samples.target),
        target              = req.target,
        model_type          = req.model_type,
        device_requested    = req.device,
        device_used         = device.type,
        )

    # Save request and response to database
    try:
        db_entry_time       = datetime.now()
        request_db_entry    = GenerateRequestDB(**req.model_dump(), time=db_entry_time)
        response_db_entry   = GenerateResponseDB(**response.model_dump(), time=db_entry_time)
        new_request_entry   = database[settings.request_collection_name].insert_one(jsonable_encoder(request_db_entry))
        new_response_entry  = database[settings.response_collection_name].insert_one(jsonable_encoder(response_db_entry))
    except errors.ServerSelectionTimeoutError as e:
        print("Could not connect to the database.")     # TODO: Log these events properly!

    return response