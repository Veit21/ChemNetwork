from fastapi import Request
from sqlmodel import Session

from app.model_registry import ModelRegistry
from app.db.engine import engine

def get_model_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry

async def get_session():
    with Session(engine) as session:
        yield session