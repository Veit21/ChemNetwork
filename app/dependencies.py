from fastapi import Request
from pymongo import MongoClient
from pymongo.synchronous.database import Database

from app.model_registry import ModelRegistry

def get_model_registry(request: Request) -> ModelRegistry:      # TODO: Look up the "Request" class and why it is useful here
    return request.app.state.registry

def get_model_database(request: Request) -> Database:
    return request.app.state.database

def get_model_mongodb_client(request: Request) -> MongoClient:
    return request.app.state.mongodb_client