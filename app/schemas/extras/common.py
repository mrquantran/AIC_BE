from typing import Generic, Optional, TypeVar
from pydantic import Field
from pydantic.generics import GenericModel

# Create a type variable that can be any type
T = TypeVar('T')

class Response(GenericModel, Generic[T]):
    data: T
    total: Optional[int] = None
    message: str = Field(example="Success", default="Success")
    status_code: int = Field( example=200, default=200)

    def __init__(self, **data):
        super().__init__(**data)
        if isinstance(self.data, (list, str, bytes)):
            self.total = len(self.data)
        else:
            self.total = None