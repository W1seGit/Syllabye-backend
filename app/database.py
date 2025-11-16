import os
from typing import Generator

from dotenv import load_dotenv
from pymongo import MongoClient, ReturnDocument
from pymongo.database import Database


load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI") or os.getenv("DATABASE_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "syllabye")

_client = MongoClient(MONGODB_URI)
_db: Database = _client[MONGODB_DB_NAME]


def get_db() -> Generator[Database, None, None]:
    try:
        yield _db
    finally:
        # No explicit close per-request; client is process-wide.
        pass


def get_next_id(collection_name: str) -> int:
    counters = _db["counters"]
    doc = counters.find_one_and_update(
        {"_id": collection_name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return int(doc["seq"])
