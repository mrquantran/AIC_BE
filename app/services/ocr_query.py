from typing import List
import asyncio
from app.common.controller import BaseController
from app.common.enum import QueryType
from app.models.ocr import OCR
from app.repositories.ocr_query import OCRQueryRepository
from app.schemas.requests.query import SearchSettings
from app.schemas.responses.keyframes import KeyframeWithConfidence
from app.config.embedding import embedder


class OCRQueryService(BaseController[OCR]):

    def __init__(self, query_repository: OCRQueryRepository):
        super().__init__(model=OCR, repository=query_repository)
        self.query_repository = query_repository

    async def search_keyframes_by_ocr(
        self,
        ocr_queries: List[str],
        settings: SearchSettings,
    ) -> List[KeyframeWithConfidence]:
        use_faiss = settings.vector_search == "faiss"

        # Perform text queries concurrently
        ocr_queries = [
            embedder.text_query(value, k=settings.k_query, use_faiss=use_faiss)
            for value in ocr_queries
        ]
        results = await asyncio.gather(*ocr_queries)

        # Flatten results and remove duplicates
        flattened_results = {
            int(idx): score for query_result in results for idx, score in query_result
        }

        ocr_indexes: list[int] = list(flattened_results.keys())

        ocr_keyframes = await self.query_repository.get_keyframe_by_indices(ocr_indexes)

        # Create KeyframeWithConfidence objects
        keyframes_with_confidence = [
            KeyframeWithConfidence(
                key=keyframe.key,
                value=keyframe.value,
                confidence=flattened_results[keyframe.key],
                type=QueryType.OCR,
            )
            for keyframe in ocr_keyframes
        ]

        return keyframes_with_confidence
