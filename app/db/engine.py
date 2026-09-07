from typing import Tuple
from pymongo import MongoClient
from pymongo.synchronous.database import Database

from app.config import settings

def init_database_connections() -> Tuple[MongoClient, Database]:
    """Initializes the connections to a MongoClient and a database specified in config.py.

        Returns:
            Tuple[MongoClient, Database]: A tuple of the client and the database objects.
    """

    # Start the client
    client = MongoClient(
        host=settings.db_uri,
        serverSelectionTimeoutMS=settings.connection_timeout_ms
        )

    # Connect to specific database
    database = client[settings.db_name]

    return client, database
