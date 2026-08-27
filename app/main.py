###############################################################
#
#   Entry point to the app.
#
###############################################################

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends

from app.model_registry import ModelRegistry
from app.config import settings
from app.routers.samples import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry.from_checkpoints(paths=settings.checkpoint_paths)
    if settings.default_target not in registry:
        raise RuntimeError(
            f"Default target '{settings.default_target.value}' has no served model. "
            f"Available: {registry.available}."
        )

    app.state.registry = registry
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(router)
app.frontend(path="/", directory="app/static")