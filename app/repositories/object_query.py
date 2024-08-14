from typing import List
from app.common.repository.base import BaseRepository
from app.models.object import Object

class ObjectQueryRepository(BaseRepository[Object]):
    """
    Query repository provides all the database operations for the Query model.
    """

    async def get_keyframe_by_object_names(self, keys: List[str]) -> List[Object]:
        """
        Get all record by indicies.

        :param keys: list of indices
        :return: A list of keyframes.
        """
        cursor = self.collection.find({"name": {"$in": keys}})
        return await cursor.to_list(length=None)

    async def get_object_names(self) -> List[str]:
        """
        Get all records from the database and return their "name" field.

        :return: A list of names from the objects in the database.
        """
        # Fetch all documents and return their "name" field
        cursor = self.collection.find({})
        result = await cursor.to_list(length=None)
        return [doc.name for doc in result]
