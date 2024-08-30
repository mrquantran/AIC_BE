from typing import Any, List
from app.common.repository.base import BaseRepository
from app.models.keyframe import Keyframe


class TextQueryRepository(BaseRepository[Keyframe]):
    """
    Query repository provides all the database operations for the Query model.
    """

    async def get_one_keyframe_by_index(self, key: int) -> List[Keyframe]:
        """
        Get one keyframe by one index

        :param key: index
        :return: A list of keyframe.
        """
        return await self.collection.find_one({"key": int(key)})

    async def get_keyframe_by_indices(self, keys: List[int]) -> List[Keyframe]:
        """
        Get all record by indicies.

        :param keys: list of indices
        :return: A list of keyframes.
        """
        cursor = self.collection.find({"key": {"$in": keys}})
        return await cursor.to_list(length=None)
