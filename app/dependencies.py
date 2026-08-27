from fastapi import Request
from app.model_registry import ModelRegistry

def get_model_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry