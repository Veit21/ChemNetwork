###############################################################
#
#   Entry point to the app.
#
###############################################################

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.model_registry import ModelRegistry
from app.config import settings
from app.routers import samples

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Create (no)SQL database and tables TODO: Implement NoSQL database with MongoDB
    # create_db_and_tables()

    # Load the models
    registry = ModelRegistry.from_checkpoints(paths=settings.checkpoint_paths)
    if settings.default_target not in registry:
        raise RuntimeError(
            f"Default target '{settings.default_target.value}' has no served model. "
            f"Available: {registry.available}."
        )
    app.state.registry = registry
    yield

    # Clean up the models and release resources
    registry.clear()

app = FastAPI(lifespan=lifespan, title=settings.name)
app.include_router(samples.router)
app.frontend(path="/", directory="app/static")

@app.get("/health", tags=["general"])
async def health():
    return{
        "status": "ok",
    }