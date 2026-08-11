import torch

from fastapi import APIRouter, Depends

from chemnetwork.sample import generate_samples
from chemnetwork.models.flow_matching import FlowModel
from app.schemas import GenerateRequest, GenerateResponse
from app.dependencies import get_model
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

    # TODO: Move this to some "app/serialization.py" for better separation of concerns?
    # Slice the trajectory for visualization purposes
    predicted_data = samples.generated
    num_frames = min(predicted_data.shape[0], settings.max_trajectory_steps)
    idx = torch.linspace(0, predicted_data.shape[0] - 1, steps=num_frames).round().long()
    predicted_data = predicted_data[idx]

    return GenerateResponse(
        num_samples=req.num_samples,
        source_points=samples.source.to(torch.float64).round(decimals=3).tolist(),
        generated_points=predicted_data.to(torch.float64).round(decimals=3).tolist()
        )