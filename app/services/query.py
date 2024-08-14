from typing import List, Tuple
from app.common.controller import BaseController
from app.common.exceptions import NotFoundException
from app.models import Text
from app.repositories import QueryRepository
from app.schemas.requests import SearchBodyRequest
from app.schemas.requests.query import SearchSettings
from app.schemas.responses.keyframes import KeyframeWithConfidence
from app.services.clip_embedding import CLIPEmbedding
import os
from app.config.embedding import embedder
import asyncio

class QueryService(BaseController[Text]):
    def __init__(self, query_repository: QueryRepository):
        super().__init__(model=Text, repository=query_repository)
        self.query_repository = query_repository

    async def get_keyframe_by_index(self, index: int) -> Text:
        result = await self.query_repository.get_one_keyframe_by_index(index)

        if not result:
            raise NotFoundException("No keyframe found")

        return result

    async def search_keyframes_by_text(
        self, body: List[SearchBodyRequest], settings: SearchSettings
    ) -> List[KeyframeWithConfidence]:
        use_faiss = settings.vector_search == 'faiss'
        print(use_faiss)

        # Perform text queries concurrently
        text_queries = [
            embedder.text_query(req.value, k=5, use_faiss=use_faiss)
            for req in body if req.model == "Text"
        ]
        results = await asyncio.gather(*text_queries)

        # Flatten results and remove duplicates
        flattened_results = {
            int(idx): score
            for query_result in results
            for idx, score in query_result
        }

        # Get keyframes for all unique indices at once
        keyframes = await self.query_repository.get_keyframe_by_indices(list(flattened_results.keys()))

        # Create KeyframeWithConfidence objects
        keyframes_with_confidence = [
            KeyframeWithConfidence(
                key=keyframe.key,
                value=keyframe.value,
                confidence=flattened_results[keyframe.key]
            )
            for keyframe in keyframes
        ]

        # Sort by confidence score in descending order
        # keyframes_with_confidence.sort(key=lambda x: x.confidence, reverse=True)

        return keyframes_with_confidence
