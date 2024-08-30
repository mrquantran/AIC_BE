from typing import List
from beanie import Document, Indexed
from pydantic import Field

class Object(Document):
    name: Indexed(str, unique=True) = Field(default="") #type: ignore
    value: List[int] = Field(default_factory=list)

    class Settings:
        collection = "objects"
        indexes = [
            "name",  # Tạo chỉ mục cho trường 'name'
        ]
