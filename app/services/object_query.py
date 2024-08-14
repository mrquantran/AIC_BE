from typing import List
from app.common.controller.base import BaseController
from app.models.object import Object
from app.repositories.object_query import ObjectQueryRepository


class ObjectQueryService(BaseController[Object]):
    def __init__(self, query_repository: ObjectQueryRepository):
        super().__init__(model=Object, repository=query_repository)
        self.query_repository = query_repository

    async def search_keyframes_by_objects(self, keys: List[str]) -> List[Object]:
        """
        Get all record by indicies.

        :param keys: list of indices
        :return: A list of keyframes.
        """
        return await self.query_repository.get_keyframe_by_object_names(keys)

    async def get_object_names(self) -> List[str]:
        """
        Get all object names.

        :return: A list of object names.
        """
        result = await self.query_repository.get_object_names()
        return result