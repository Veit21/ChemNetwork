from fastapi import APIRouter, Depends, HTTPException

from chemnetwork.sample import generate_samples
from chemnetwork.models.flow_matching import FlowModel
from app.schemas import GenerateRequest, GenerateResponse
from app.dependencies import get_model_registry
from app.serialization import downsample_trajectory_tensor, typecast_and_round_output
from app.config import settings
from app.model_registry import ModelRegistry

router = APIRouter()

@router.get("/health")
def health(registry: ModelRegistry = Depends(get_model_registry)):
    return{
        "status": "ok",
        "available_targets": registry.available,
    }

@router.post("/generate", response_model=GenerateResponse)
def generate(
    req: GenerateRequest,
    registry: ModelRegistry = Depends(get_model_registry)
    ) -> GenerateResponse:
    """API endpoint to generate samples from the learned target distribution p_1.

    Args:
        req (GenerateRequest): Request body containing the number of samples and integration steps.

    Returns:
        GenerateResponse: The response containing the generated samples.
    """
    try:
        entry = registry.get(req.target)
    except KeyError:
        raise HTTPException(404, f"No model served for '{req.target}'. Available: {registry.available}.")

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
        target_name=req.target
        )