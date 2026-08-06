###############################################################
#
#   Entry point to the app.
#
###############################################################

from contextlib import asynccontextmanager
from fastapi import FastAPI

from chemnetwork.sample import load_model
from app.config import settings
from app.routers.samples import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(checkpoint_path=settings.checkpoint_path)
    yield

@router.get("/health")
def health():
    return{"status": "ok"}

app = FastAPI(lifespan=lifespan)
app.include_router(router)