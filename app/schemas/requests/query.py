from typing import Union, List
from pydantic import BaseModel, Field
from app.common.enum import QueryType


class TemporalVideoQuery(BaseModel):
    video_id: int
    keyframes: List[int] 


class TemporalGroupQuery(BaseModel):
    group_id: int 
    videos: List[TemporalVideoQuery]


class SearchBodyRequest(BaseModel):
    model: QueryType = Field(default="Text", example="Text")
    value: Union[str, List[str], List[TemporalGroupQuery]] = Field(
        default="Text", example="HCM AI"
    )


class GetNearestIndexRequest(BaseModel):
    group_id: int
    video_id: int
    keyframe_id: int

class SearchSettings(BaseModel):
    vector_search: str = Field(default="faiss")
    k_query: int = Field(default=5)
    display: int = Field(default=5)
