from typing import Any, Dict, List, Tuple
import asyncio
from app.common.controller import BaseController
from app.common.enum import QueryType
from app.models import Keyframe
from app.repositories import TextQueryRepository
from app.schemas.requests.query import SearchSettings, TemporalGroupQuery
from app.schemas.responses.keyframes import KeyFrameInformation, KeyframeWithConfidence
from app.config.embedding import embedder
from app.services.ocr_query import OCRQueryService


class TextQueryService(BaseController[Keyframe]):

    def __init__(self, query_repository: TextQueryRepository):
        super().__init__(model=Keyframe, repository=query_repository)
        self.query_repository = query_repository

    async def get_nearest_index(
        self, group_id: int, video_id: int, keyframe_id: int
    ) -> KeyFrameInformation:
        return await self.query_repository.get_index_by_keyframe_information(
            group_id=group_id, video_id=video_id, keyframe_id=keyframe_id
        )

    def extract_keyframe_tuples(
        self, mongodb_results: List[dict], temporal_groups: List[TemporalGroupQuery]
    ) -> List[Tuple[int, int]]:
        # Create a lookup dictionary from MongoDB results for quick access to max keyframe values
        max_keyframe_lookup = {
            (result["video_id"], result["group_id"]): max(result["range"])
            for result in mongodb_results
        }

        output = []

        for group in temporal_groups:
            for video in group.videos:
                keyframes = video.keyframes

                if len(keyframes) > 1:
                    # More than one keyframe, return (min, max) tuple
                    output.append((min(keyframes), max(keyframes)))
                elif len(keyframes) == 1:
                    # Only one keyframe, find the max keyframe from the corresponding MongoDB result
                    max_keyframe_of_video = max_keyframe_lookup.get(
                        (video.video_id, group.group_id), None
                    )
                    if max_keyframe_of_video is not None:
                        output.append((keyframes[0], max_keyframe_of_video))

        return output

    async def search_range_by_groups(
        self, groups_videos_queries: List[TemporalGroupQuery]
    ) -> List[Tuple[int, int]]:
        groups_list = [group.group_id for group in groups_videos_queries]
        video_list = [
            video.video_id for group in groups_videos_queries for video in group.videos
        ]

        # query the max and min keyframe index of each video
        keyframe_by_group_video = (
            await self.query_repository.get_max_min_keyframe_by_video_and_group(
                groups_list, video_list
            )
        )

        print(f"Keyframe by group video: {keyframe_by_group_video}")

        result = self.extract_keyframe_tuples(
            keyframe_by_group_video, groups_videos_queries
        )
        print(f"Result: {result}")

        return result

    async def search_keyframes_by_text(
        self,
        text_queries: List[str],
        object_queries: Tuple[List[str], List[int]],
        settings: SearchSettings,
        audio_queries: List[str],
        range_queries: List[Tuple[int, int]],
        ocr_queries: List[str],
    ) -> Tuple[List[Keyframe], List[Keyframe]]:
        # from settings query params
        use_faiss = settings.vector_search == "faiss"
        kquery = settings.k_query
        audio_results = []
        text_results = []

        # Unpack object queries
        object_tags_query, object_indexes = object_queries
        print(f"Object tags: {object_tags_query}")

        audio_queries = [
            embedder.audio_query_by_text(text_query=value, k=kquery)
            for value in audio_queries
        ]
        print(f"Audio queries: {audio_queries}")
    
        if len(audio_queries) > 0:
            audio_results = await asyncio.gather(*audio_queries)
            print(f"Audio results: {audio_results}")

        text_queries = [
            embedder.text_query(
                value, k=kquery, use_faiss=use_faiss, ranges=range_queries
            )
            for value in text_queries
        ]
        if len(text_queries) > 0:
            text_results = await asyncio.gather(*text_queries)
            print(f"Text results: {text_results}")

        flattened_results_audio = {
            int(idx): score for query_result in audio_results for idx, score in query_result
        }

        # Flatten results and remove duplicates
        flattened_results = {
            int(idx): score
            for query_result in text_results
            for idx, score in query_result
        }

        text_indexes: list[int] = list(flattened_results.keys())

        # Get keyframes for all unique indices at once
        # Perform database fetch operations concurrently using asyncio.gather
        text_keyframes_task = self.query_repository.get_keyframe_by_indices(
            text_indexes
        )
        object_keyframes_task = self.query_repository.get_keyframe_by_indices(
            object_indexes,
        )
        audio_query_task = self.query_repository.get_keyframe_by_audio_indexes(
            list(flattened_results_audio.keys())
        )

        text_keyframes, object_keyframes, audio_keyframes = await asyncio.gather(
            text_keyframes_task, object_keyframes_task, audio_query_task
        )

        keyframes_with_confidence = [
            KeyframeWithConfidence(
                key=keyframe.key,
                value=keyframe.value,
                confidence=flattened_results[keyframe.key],
                video_id=keyframe.video_id,
                group_id=keyframe.group_id,
            )
            for keyframe in text_keyframes
        ]

        keyframes_audio = [
            KeyframeWithConfidence(
                key=keyframe.key,
                value=keyframe.value,
                confidence=flattened_results_audio[keyframe.audio_index],
                video_id=keyframe.video_id,
                group_id=keyframe.group_id,
            )
            for keyframe in audio_keyframes
        ]

        keyframes_with_object = [
            KeyframeWithConfidence(
                key=keyframe.key,
                value=keyframe.value,
                confidence=[
                    keyframe.tags.get(object_tag)
                    for object_tag in object_tags_query
                    if keyframe.tags.get(object_tag) is not None
                ],
                video_id=keyframe.video_id,
                group_id=keyframe.group_id,
            )
            for keyframe in object_keyframes
        ]

        keyframes_with_object_splitted = [
            KeyframeWithConfidence(
                key=keyframe.key,
                value=keyframe.value,
                confidence=confidence,
                video_id=keyframe.video_id,
                group_id=keyframe.group_id,
            )
            for keyframe in keyframes_with_object
            for confidence in keyframe.confidence
        ]

        return (
            keyframes_with_confidence,
            keyframes_with_object_splitted,
            keyframes_audio,
        )
