from typing import List
from app.common.repository.base import BaseRepository
from app.models.ocr import OCR


class OCRQueryRepository(BaseRepository[OCR]):
    """
    Query repository provides all the database operations for the Query model.
    """

    async def get_keyframe_by_indices(self, keys: List[int]) -> List[OCR]:
        """
        Get all record by indicies.

        :param keys: list of indices
        :return: A list of keyframes.
        """
        print("keys", keys)
        # conert key to int
        keys = [int(key) for key in keys]
        cursor = self.collection.find({"key": {"$in": keys}})
        return await cursor.to_list(length=None)
