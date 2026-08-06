from fastapi import Request
from chemnetwork.models.flow_matching import FlowModel

def get_model(request: Request) -> FlowModel:
    return request.app.state.model