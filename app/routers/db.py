from typing import Annotated, List, Dict
from fastapi import APIRouter, Depends

from pymongo.synchronous.database import Database

from app.dependencies import get_model_database
from app.schemas import GenerateRequestDB


# Set router parameters
router = APIRouter(
    prefix="/db",
    tags=["db"],
)

# TODO: One endpoint per collection? Or keep with path parameter?
@router.get("/read_all/{collection_name}", response_model=List[GenerateRequestDB])
def read_all_db_entries(collection_name: str, database: Annotated[Database, Depends(get_model_database)]) -> List[GenerateRequestDB]:
    return list(database[collection_name].find(limit=100))

# TODO: Somehow possible to "hide" this endpoint? Or secure it?
@router.get("/clear_all/{collection_name}")
def clear_all_db_entries(collection_name: str, database: Annotated[Database, Depends(get_model_database)]) -> Dict:
    """DENGAROUS! Definitely delete later. Or "make private" or so.
    """
    for doc in database[collection_name].find():
        database[collection_name].delete_one(doc)
    return {
        "message": "database cleared!"
    }