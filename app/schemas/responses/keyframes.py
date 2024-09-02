from typing import Optional
from pydantic import BaseModel, Field

from app.common.enum import QueryType

class KeyframeWithConfidence(BaseModel):
    key: int
    value: str
    confidence: Optional[float] | list[float]
    video_id: int
    group_id: int
    # type: QueryType = QueryType.TEXT

class KeyFrameInformation(BaseModel):
    index: int
    frame_id: int
    video_id: int
    group_id: int
    value: str
    # type: QueryType = QueryType.TEXT