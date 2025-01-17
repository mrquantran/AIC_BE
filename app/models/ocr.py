from typing import Optional
from beanie import Document, Indexed
from pydantic import Field


class OCR(Document):
    # Beanie automatically uses ObjectId for the '_id' field
    key: Indexed(int, unique=True) = Field(default=0)  # type: ignore
    value: Optional[str] = None
    video_id: Indexed(int) = Field(default=0) # type: ignore
    group_id: Indexed(int) = Field(default=0) # type: ignore 
    frame_id: Indexed(int) = Field(default=0) # type: ignore

    class Settings:
        indexes = [
            "key",  # This creates a single-field index on the 'key' field
        ]
