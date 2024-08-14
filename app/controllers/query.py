from typing import List, Tuple
from app.models import Text
from app.schemas.requests import SearchBodyRequest, SearchSettings
from app.services import QueryService


class QueryController:
    def __init__(self, query_serivce: QueryService):
        self.query_serivce = query_serivce

    async def get_keyframe_by_index(self, index: int) -> Text:
        # get all method in repository
        return await self.query_serivce.get_keyframe_by_index(index)

    async def search_keyframes_by_text(
        self, body: List[SearchBodyRequest], settings: SearchSettings
    ):
        return await self.query_serivce.search_keyframes_by_text(body, settings)
