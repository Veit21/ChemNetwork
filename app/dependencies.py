from fastapi import Request
from chemnetwork.models.flow_matching import FlowModel

def get_model(request: Request) -> FlowModel:
    return request.app.state.loaded_model.model

def get_model_config(request: Request) -> dict:
    return request.app.state.loaded_model.config