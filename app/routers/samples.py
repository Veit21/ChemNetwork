from fastapi import APIRouter, Depends, HTTPException

from chemnetwork.sample import generate_samples
from chemnetwork.data.point_clouds import TargetDistribution
from app.schemas import AvailableResponse, GenerateRequest, GenerateResponse, TargetInfo
from app.dependencies import get_model_registry
from app.serialization import downsample_trajectory_tensor, typecast_and_round_output
from app.config import settings
from app.model_registry import ModelRegistry

router = APIRouter()

@router.get("/health")
def health(registry: ModelRegistry = Depends(get_model_registry)):
    return{
        "status": "ok",
    }

@router.get("/available", response_model=AvailableResponse)
def available(registry: ModelRegistry = Depends(get_model_registry)) -> AvailableResponse:
    """API endpoint to retrieve the available learned target distributions.

    Args:
        registry (ModelRegistry, optional): Registry object that manages the loaded models and its properties. Defaults to Depends(get_model_registry).

    Returns:
        AvailableResponse: Every served target with its display label, plus the target
        the frontend should preselect.
    """
    return AvailableResponse(
        targets=[TargetInfo.from_target(TargetDistribution(target)) for target in registry.available],
        default=settings.default_target,
    )

@router.post("/generate", response_model=GenerateResponse)
def generate(
    req: GenerateRequest,
    registry: ModelRegistry = Depends(get_model_registry)
    ) -> GenerateResponse:
    """API endpoint to generate samples from the learned target distribution p_1.

    An unknown target name is rejected by the request schema with a 422; a known
    target with no served checkpoint is rejected here with a 404.

    Args:
        req (GenerateRequest): Request body containing the number of samples, integration steps and the target distribution.

    Returns:
        GenerateResponse: The response containing the generated samples.
    """
    try:
        # Retrieve the model that generates the desired target data
        entry = registry.get(req.target)
    except KeyError:
        raise HTTPException(404, f"No model served for '{req.target.value}'. Available: {registry.available}.")

    # Generate samples
    samples = generate_samples(
        model=entry.model,
        cfg=entry.config,
        num_samples=req.num_samples,
        integration_steps=req.integration_steps,
        return_trajectory=req.return_trajectory,
    )
    
    # Subsample the trajectory
    predicted_data_subsampled = downsample_trajectory_tensor(
        trajectory=samples.generated,
        max_steps=settings.max_trajectory_steps
    )

    return GenerateResponse(
        num_samples=req.num_samples,
        source_points=typecast_and_round_output(samples.source),
        generated_points=typecast_and_round_output(predicted_data_subsampled),
        target_points=typecast_and_round_output(samples.target),
        target=req.target
        )