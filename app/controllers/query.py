from typing import List
from itertools import chain
import asyncio

from app.common.enum import QueryType
from app.schemas.requests import SearchBodyRequest, SearchSettings
from app.services import TextQueryService
from app.services.object_query import ObjectQueryService
from app.services.reciprocal_rank_fusion import ReciporalRankFusionService
from itertools import chain


class QueryController:
    def __init__(
        self,
        text_query_serivce: TextQueryService,
        object_query_service: ObjectQueryService,
        ocr_query_service: ObjectQueryService,
    ):
        self.text_query_serivce = text_query_serivce
        self.object_query_service = object_query_service
        self.ocr_query_service = ocr_query_service

    async def search_keyframes(
        self, body: List[SearchBodyRequest], settings: SearchSettings
    ):
        text_queries = []
        object_tags = []

        for req in body:
            if req.model == QueryType.TEXT:
                text_queries.append(req.value)
            elif req.model == QueryType.OBJECT:
                object_tags.append(req.value)

        # filter the text queries
        text_queries = list(text_queries)

        # filter by ocr queries
        # ocr_queries = [req.value for req in body if req.model == QueryType.OCR]

        object_tags = list(chain.from_iterable(object_tags))
        object_query_index = []

        if len(object_tags) > 0:
            object_query_results = (
                await self.object_query_service.search_keyframes_by_objects(object_tags)
            )
            object_query_index: List[int] = [
                object_item.value for object_item in object_query_results
            ]
            object_query_index = list(set(chain.from_iterable(object_query_index)))

        object_query = (
            object_tags,
            object_query_index,
        )

        clip_keyframes, object_keyframes = (
            await self.text_query_serivce.search_keyframes_by_text(
                text_queries, object_query, settings
            )
        )

        reciporalRankFusionService = ReciporalRankFusionService()

        results = reciporalRankFusionService.reciprocal_rank_fusion(
            clip_results=reciporalRankFusionService.format_keyframes_results(
                clip_keyframes
            ),
            object_results=reciporalRankFusionService.format_keyframes_results(
                object_keyframes
            ),
        )

        final_results = reciporalRankFusionService.combine_results(
            clip_keyframes + object_keyframes, results
        )[0:settings.k_query]

        return final_results
