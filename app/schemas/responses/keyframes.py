from typing import Optional
from pydantic import BaseModel, Field

from app.common.enum import QueryType

class KeyframeWithConfidence(BaseModel):
    key: int
    value: str
    confidence: Optional[float]
    type: QueryType = Field(default=QueryType.TEXT, example=QueryType.TEXT)
