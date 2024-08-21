from typing import List
from itertools import chain
import asyncio

from app.common.enum import QueryType
from app.schemas.requests import SearchBodyRequest, SearchSettings
from app.services import TextQueryService
from app.services.object_query import ObjectQueryService


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
        # filter the text queries
        text_queries = [req.value for req in body if req.model == QueryType.TEXT]

        # filter by ocr queries
        ocr_queries = [req.value for req in body if req.model == QueryType.OCR]

        object_queries = list(
            chain.from_iterable(
                [req.value for req in body if req.model == QueryType.OBJECT]
            )
        )
        object_query_index = []

        if len(object_queries) > 0:
            object_query_results = (
                await self.object_query_service.search_keyframes_by_objects(
                    object_queries
                )
            )
            object_query_index: List[int] = [
                object_item.value for object_item in object_query_results
            ]
            object_query_index = list(set(chain.from_iterable(object_query_index)))

        ocr_keyframes_task = self.ocr_query_service.search_keyframes_by_ocr(
            ocr_queries, settings
        )
        text_object_keyframes_task = self.text_query_serivce.search_keyframes_by_text(
            text_queries, object_query_index, settings
        )
        text_object_keyframes, object_keyframes = await asyncio.gather(
            text_object_keyframes_task, ocr_keyframes_task
        )

        return text_object_keyframes + object_keyframes
