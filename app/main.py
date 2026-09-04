###############################################################
#
#   Entry point to the app.
#
###############################################################

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pymongo import MongoClient, errors

from app.model_registry import ModelRegistry
from app.config import settings
from app.routers import samples, db

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Connect to database
    is_connected_to_db = False
    try:
        app.state.mongodb_client = MongoClient(     # NOTE: Why app.state? why not omit the .state? See https://www.mongodb.com/resources/languages/pymongo-tutorial for more
            host=settings.db_uri,
            serverSelectionTimeoutMS=3000,
        )
        app.state.database = app.state.mongodb_client[settings.db_name]
        is_connected_to_db = True

    except errors.ServerSelectionTimeoutError as e:
        print(f"Couldn't connect to the database.")    # TODO: This happens 'silently' for the client! Fix?

    # Load the models
    registry = ModelRegistry.from_checkpoints(paths=settings.checkpoint_paths)
    if settings.default_target not in registry:
        raise RuntimeError(
            f"Default target '{settings.default_target.value}' has no served model. "
            f"Available: {registry.available}."
        )
    app.state.registry = registry
    yield

    # Disconnect from database
    if is_connected_to_db:
        app.state.mongodb_client.close()

    # Clean up the models and release resources
    app.state.registry.clear()

app = FastAPI(lifespan=lifespan, title=settings.name)
app.include_router(samples.router)
app.include_router(db.router)
app.frontend(path="/", directory="app/static")

@app.get("/health", tags=["general"])
async def health():
    return{
        "status": "ok",
    }