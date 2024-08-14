from pydantic import BaseModel

class KeyframeWithConfidence(BaseModel):
    key: int
    value: str
    confidence: float