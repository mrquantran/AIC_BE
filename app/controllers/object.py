from typing import List
from app.services.object_query import ObjectQueryService


class ObjectController:
    def __init__(
        self,
        object_query_service: ObjectQueryService,
    ):
        self.object_query_service = object_query_service


    async def get_object_names(self) -> List[str]:
        return await self.object_query_service.get_object_names()