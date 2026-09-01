from typing import Annotated
from fastapi import APIRouter, Depends
from sqlmodel import Session

# from app.db.engine import *
from app.dependencies import get_session

router = APIRouter(
    prefix="/db",
    tags=["db"],
)

# SessionDep = Annotated[Session, Depends(get_session)]