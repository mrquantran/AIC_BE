from typing import List, Tuple
import asyncio
from app.common.controller import BaseController
from app.common.enum import QueryType
from app.models import Keyframe
from app.repositories import TextQueryRepository
from app.schemas.requests.query import SearchSettings
from app.schemas.responses.keyframes import KeyframeWithConfidence
from app.config.embedding import embedder


class TextQueryService(BaseController[Keyframe]):

    def __init__(self, query_repository: TextQueryRepository):
        super().__init__(model=Keyframe, repository=query_repository)
        self.query_repository = query_repository

    async def search_keyframes_by_text(
        self,
        text_queries: List[str],
        object_queries: Tuple[List[str], List[int]],
        settings: SearchSettings,
    ) -> Tuple[List[Keyframe], List[Keyframe]]:
        # Unpack object queries
        # object_tags_query is a list of object tags
        # object_indexes is a list of object indices existed queried in previous step
        object_tags_query, object_indexes = object_queries
        print(f"Object tags: {object_tags_query}")

        # from settings query params
        use_faiss = settings.vector_search == "faiss"
        kquery = settings.k_query

        # Perform Keyframe queries concurrently
        text_queries = [
            embedder.text_query(value, k=kquery, use_faiss=use_faiss)
            for value in text_queries
        ]
        results = await asyncio.gather(*text_queries)

        # Flatten results and remove duplicates
        flattened_results = {
            int(idx): score for query_result in results for idx, score in query_result
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

        text_keyframes, object_keyframes = await asyncio.gather(
            text_keyframes_task, object_keyframes_task
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

        return (keyframes_with_confidence, keyframes_with_object_splitted)
