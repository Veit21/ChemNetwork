from fastapi import APIRouter, Depends

from chemnetwork.sample import generate_samples
from chemnetwork.models.flow_matching import FlowModel
from app.schemas import GenerateRequest, GenerateResponse
from app.dependencies import get_model

router = APIRouter()

@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, model: FlowModel = Depends(get_model)):
    samples = generate_samples(
        model=model,
        num_samples=req.num_samples,
        integration_steps=req.integration_steps
    )
    return GenerateResponse(num_samples=req.num_samples, points=samples.tolist())