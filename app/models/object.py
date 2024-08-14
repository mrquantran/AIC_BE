from typing import List
from beanie import Document, Indexed
from pydantic import Field


class Object(Document):
    name: Indexed(str, unique=True) = Field(default=0)  # type: ignore
    value: List[int] = []

    class Settings:
        indexes = [
            "key",  # This creates a single-field index on the 'key' field
        ]
