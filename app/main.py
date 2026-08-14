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
    app.state.registry = ModelRegistry.from_checkpoints(paths=settings.checkpoint_paths)
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(router)
app.frontend(path="/", directory="app/static")