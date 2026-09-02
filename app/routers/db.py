from typing import Annotated
from fastapi import APIRouter, Depends
from sqlmodel import Session

# from app.db.engine import *

router = APIRouter(
    prefix="/db",
    tags=["db"],
)

# SessionDep = Annotated[Session, Depends(get_session)]