from typing import List
import asyncio
from app.common.controller import BaseController
from app.common.enum import QueryType
from app.models import Text
from app.repositories import TextQueryRepository
from app.schemas.requests.query import SearchSettings
from app.schemas.responses.keyframes import KeyframeWithConfidence
from app.config.embedding import embedder

class TextQueryService(BaseController[Text]):

    def __init__(self, query_repository: TextQueryRepository):
        super().__init__(model=Text, repository=query_repository)
        self.query_repository = query_repository

    async def search_keyframes_by_text(
        self, text_queries: List[str], object_queries: List[int], settings: SearchSettings
    ) -> List[KeyframeWithConfidence]:
        use_faiss = settings.vector_search == 'faiss'

        # Perform text queries concurrently
        text_queries = [
            embedder.text_query(value, k=settings.k_query, use_faiss=use_faiss)
            for value in text_queries
        ]
        results = await asyncio.gather(*text_queries)

        # Flatten results and remove duplicates
        flattened_results = {
            int(idx): score
            for query_result in results
            for idx, score in query_result
        }

        text_indexes: list[int] = list(flattened_results.keys())
        object_indexes: list[int] = object_queries

        # Get keyframes for all unique indices at once
        # Perform database fetch operations concurrently using asyncio.gather
        text_keyframes_task = self.query_repository.get_keyframe_by_indices(
            text_indexes
        )
        object_keyframes_task = self.query_repository.get_keyframe_by_indices(
            object_indexes
        )

        text_keyframes, object_keyframes = await asyncio.gather(
            text_keyframes_task, object_keyframes_task
        )
        # Create KeyframeWithConfidence objects
        keyframes_with_confidence = [
            KeyframeWithConfidence(
                key=keyframe.key,
                value=keyframe.value,
                confidence=flattened_results[keyframe.key],
                type=QueryType.TEXT,
            )
            for keyframe in text_keyframes
        ]

        keyframes_by_object = [
            KeyframeWithConfidence(
                key=keyframe.key,
                value=keyframe.value,
                confidence=None,
                type=QueryType.OBJECT,
            )
            for keyframe in object_keyframes
        ]

        # Sort by confidence score in descending order
        # keyframes_with_confidence.sort(key=lambda x: x.confidence, reverse=True)

        return keyframes_with_confidence + keyframes_by_object
