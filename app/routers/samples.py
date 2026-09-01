from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from sqlmodel import Session
from time import strftime, localtime

from chemnetwork.sample import generate_samples
from chemnetwork.data.point_clouds import TargetDistribution
from app.schemas import AvailableResponse, GenerateRequest, GenerateResponse, TargetInfo, RequestDB
from app.dependencies import get_model_registry, get_session
from app.serialization import downsample_trajectory_tensor, typecast_and_round_output
from app.config import settings
from app.model_registry import ModelRegistry


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
    session: Annotated[Session, Depends(get_session)]
) -> GenerateResponse:
    """ API endpoint to generate samples from the learned target distribution p_1.

        Args:
            req (GenerateRequest): Request body containing the number of samples, integration steps and the target distribution.

        Returns:
            GenerateResponse: The response containing the generated samples.
    """
    try:
        # Retrieve the model that generates the desired target data
        entry = registry.get(req.target)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model served for '{req.target.value}'. Available: {registry.available}."
        )   

    # Generate samples
    samples = generate_samples(
        model               = entry.model,
        cfg                 = entry.config,
        num_samples         = req.num_samples,
        integration_steps   = req.integration_steps,
        return_trajectory   = req.return_trajectory,
    )
    
    # Subsample the trajectory
    # TODO: Subsampling still breaks if return_trajectory=False. Maybe always make the network return the trajectory.
    predicted_data_subsampled = downsample_trajectory_tensor(
        trajectory  = samples.generated,
        max_steps   = settings.max_trajectory_steps
    )

    # Save request to DB
    db_req = RequestDB(
        num_samples=req.num_samples,
        integration_steps=req.integration_steps,
        return_trajectory=req.return_trajectory,
        target=req.target,
        timestamp=strftime('%Y-%m-%d_%H:%M:%S', localtime())
    )
    session.add(db_req)
    session.commit()
    session.refresh(db_req)

    return GenerateResponse(
        num_samples         = req.num_samples,
        source_points       = typecast_and_round_output(samples.source),
        generated_points    = typecast_and_round_output(predicted_data_subsampled),
        target_points       = typecast_and_round_output(samples.target),
        target              = req.target
        )