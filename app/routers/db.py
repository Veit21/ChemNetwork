from typing import Annotated, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status

from pymongo import errors
from pymongo.synchronous.database import Database

from app.dependencies import get_model_database
from app.schemas import GenerateRequestDB, GenerateResponseDB
from app.config import settings


# Set router parameters
router = APIRouter(
    prefix="/db",
    tags=["db"],
)

@router.get(f"/read_all/{settings.request_collection_name}", response_model=List[GenerateRequestDB])
def read_all_db_entries_requests(limit: int, database: Annotated[Database, Depends(get_model_database)]) -> List[GenerateRequestDB]:
    try:
        return list(database[settings.request_collection_name].find(limit=limit))
    except errors.ServerSelectionTimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The database is not reachable."
        )

@router.get(f"/read_all/{settings.response_collection_name}", response_model=List[GenerateResponseDB])
def read_all_db_entries_responses(limit: int, database: Annotated[Database, Depends(get_model_database)]) -> List[GenerateResponseDB]:
    try:
        return list(database[settings.response_collection_name].find(limit=limit))
    except errors.ServerSelectionTimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The database is not reachable."
        )

@router.get(f"/{settings.request_collection_name}" + "/{id}", response_model=GenerateRequestDB)
def read_request_entry_by_id(id: str, database: Annotated[Database, Depends(get_model_database)]) -> GenerateRequestDB:
    try:
        entry = database[settings.request_collection_name].find_one({"_id": id})
        if entry is not None:
            return entry
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entry with {id=} not found."
        )
    except errors.ServerSelectionTimeoutError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The database is not reachable."
            )

@router.get(f"/{settings.response_collection_name}" + "/{id}", response_model=GenerateResponseDB)
def read_response_entry_by_id(id: str, database: Annotated[Database, Depends(get_model_database)]) -> GenerateResponseDB:
    try:
        entry = database[settings.response_collection_name].find_one({"_id": id})
        if entry is not None:
            return entry
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entry with {id=} not found."
        )
    except errors.ServerSelectionTimeoutError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The database is not reachable."
            )

@router.get(f"/query/{settings.request_collection_name}", response_model=List[GenerateRequestDB])
def read_requests_by_query(field_name: str, value: Any, database: Annotated[Database, Depends(get_model_database)]) -> List[GenerateRequestDB]:
    try:
        return list(database[settings.request_collection_name].find({field_name: value}))
    except errors.ServerSelectionTimeoutError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The database is not reachable."
            )
    
@router.get(f"/query/{settings.response_collection_name}", response_model=List[GenerateResponseDB])
def read_responses_by_query(field_name: str, value: Any, database: Annotated[Database, Depends(get_model_database)]) -> List[GenerateResponseDB]:
    try:
        return list(database[settings.response_collection_name].find({field_name: value}))
    except errors.ServerSelectionTimeoutError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The database is not reachable."
            )

@router.delete("/clear_all/{collection_name}")
def clear_all_db_entries(collection_name: str, database: Annotated[Database, Depends(get_model_database)]) -> Dict:
    """DENGAROUS! Delete later, or "secure" it somehow.
    """
    try:
        for doc in database[collection_name].find():
            database[collection_name].delete_one(doc)
        return {
            "message": "database cleared!"
        }
    except errors.ServerSelectionTimeoutError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The database is not reachable."
            )