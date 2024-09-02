from typing import Any, Dict, List
from app.common.repository.base import BaseRepository
from app.models.keyframe import Keyframe
from app.schemas.responses.keyframes import KeyFrameInformation


class TextQueryRepository(BaseRepository[Keyframe]):
    """
    Query repository provides all the database operations for the Query model.
    """

    async def get_max_min_keyframe_by_video_and_group(
        self, groups: List[int], videos: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Get max and min keyframe by video and group.

        :param groups: list of groups
        :param videos: list of videos
        :return: A list of dictionaries, each containing a video_id and the range of keyframes [min_key, max_key].
        """
        pipeline = [
            {
                "$match": {
                    "group_id": {"$in": groups},
                    "video_id": {"$in": videos},
                }
            },
            {
                "$group": {
                    "_id": {"video_id": "$video_id", "group_id": "$group_id"},
                    "min_key": {"$min": "$key"},
                    "max_key": {"$max": "$key"},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "video_id": "$_id.video_id",
                    "group_id": "$_id.group_id",
                    "range": ["$min_key", "$max_key"],
                }
            },
        ]

        cursor = self.collection.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def get_index_by_keyframe_information(
        self, group_id: int, video_id: int, keyframe_id: int
    ):
        """
        Query in the database to based on group_id, video_id
        get the value the nearest the keyframe_id
        for example the frame_id = 5, the keyframe in database only have [4, 8, 10] the result will be 4 (key field)
        or the frame_id = 9, the keyframe in database only have [4, 8, 10] the result will be 10 (key field)

        the output should be key field but the calculate is based on frame_id
        """
        pipeline = [
            {
                "$match": {
                    "group_id": group_id,
                    "video_id": video_id,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "key": 1,
                    "frame_id": 1,
                    "value": 1,
                }
            },
            {
                "$sort": {"frame_id": 1}
            }   
        ]

        cursor = self.collection.aggregate(pipeline)
        keyframes = await cursor.to_list(length=None)
        # use the abs to get the nearest keyframe
        keyframe = min(keyframes, key=lambda x: abs(x["frame_id"] - keyframe_id))
        return KeyFrameInformation(
            index=keyframe["key"],
            frame_id=keyframe["frame_id"],
            value=keyframe["value"],
            video_id=video_id,
            group_id=group_id,
        )

    async def get_keyframe_by_indices(self, keys: List[int]) -> List[Keyframe]:
        """
        Get all record by indicies.

        :param keys: list of indices
        :return: A list of keyframes.
        """
        cursor = self.collection.find({"key": {"$in": keys}})
        return await cursor.to_list(length=None)
