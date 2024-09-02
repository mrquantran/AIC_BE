from typing import List
from itertools import chain
import asyncio

from app.common.enum import QueryType
from app.schemas.requests import SearchBodyRequest, SearchSettings
from app.schemas.requests.query import TemporalGroupQuery
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

    async def get_nearest_index(
        self, group_id: int, video_id: int, keyframe_id: int
    ):
        return self.text_query_serivce.get_nearest_index(
            group_id=group_id, video_id=video_id, keyframe=keyframe_id
        )

    async def search_keyframes(
        self, body: List[SearchBodyRequest], settings: SearchSettings
    ):
        text_queries = []
        temporal_queries = []
        object_tags = []

        for req in body:
            if req.model == QueryType.TEXT:
                text_queries.append(req.value)
            elif req.model == QueryType.OBJECT:
                object_tags.append(req.value)
            elif req.model == QueryType.TEMPORAL:
                temporal_queries.append(req.value)
        print(f"temporal ${temporal_queries}")
        groups_videos_queries: List[TemporalGroupQuery] = []
        keyframe_by_group_video = []
        if len(temporal_queries) > 0:
            print(f"temporal ${temporal_queries}")
            # auto get the first element of temporal queries
            groups_videos_queries = temporal_queries[0]

            keyframe_by_group_video = (
                await self.text_query_serivce.search_range_by_groups(
                    groups_videos_queries
                )
            )

        # filter the text queries
        clip_query = list(text_queries)

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
                text_queries=clip_query,
                object_queries=object_query,
                settings=settings,
                range_queries=keyframe_by_group_video,
            )
        )

        results = ReciporalRankFusionService().reciprocal_rank_fusion(
            clip_keyframes,
            object_keyframes,
        )[0 : settings.display]

        return results
