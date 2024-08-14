from pydantic import BaseModel, Field
from app.common.enum import QueryType


class SearchBodyRequest(BaseModel):
    model: QueryType = Field(default="Text", example="Text")
    value: str = Field(default="Text", example="HCM AI")


class SearchSettings(BaseModel):
    vector_search: str = Field(default="faiss")
