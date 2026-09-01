from sqlmodel import SQLModel, create_engine

from app.config import settings


# Create engines and tables
connect_args = {"check_same_thread": False}
engine = create_engine(url=settings.db_url, connect_args=connect_args)
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)