from typing import List
from app.common.controller.base import BaseController
from app.models.object import Object
from app.repositories.object_query import ObjectQueryRepository


class ObjectQueryService(BaseController[Object]):
    def __init__(self, query_repository: ObjectQueryRepository):
        super().__init__(model=Object, repository=query_repository)
        self.query_repository = query_repository

    async def get_keyframe_by_indices(self, keys: List[str]) -> List[Object]:
        """
        Get all record by indicies.

        :param keys: list of indices
        :return: A list of keyframes.
        """
        cursor = self.collection.find({"name": {"$in": keys}})
        return await cursor.to_list(length=None)