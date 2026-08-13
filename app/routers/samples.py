from fastapi import APIRouter, Depends

from chemnetwork.sample import generate_samples
from chemnetwork.models.flow_matching import FlowModel
from app.schemas import GenerateRequest, GenerateResponse
from app.dependencies import get_model
from app.serialization import downsample_trajectory_tensor, typecast_and_round_output
from app.config import settings

router = APIRouter()

@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, model: FlowModel = Depends(get_model)) -> GenerateResponse:
    """API endpoint to generate samples from the learned target distribution p_1.

    Args:
        req (GenerateRequest): Request body containing the number of samples and integration steps.
        model (FlowModel, optional): The flow model for generating samples. Defaults to Depends(get_model).

    Returns:
        GenerateResponse: The response containing the generated samples.
    """

    # Generate samples
    samples = generate_samples(
        model=model,
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
        generated_points=typecast_and_round_output(predicted_data_subsampled)
        )