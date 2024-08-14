from typing import List
from app.common.factory import Factory
from app.controllers.object import ObjectController
from app.schemas.responses.keyframes import KeyframeWithConfidence
from fastapi import APIRouter, Body, Depends, Query
from app.schemas.extras import Response
from app.schemas.requests import SearchBodyRequest, SearchSettings
from app.services import TextQueryService
from app.controllers import QueryController
from app.services.object_query import ObjectQueryService

query_router = APIRouter()
factory = Factory()


def get_query_controller(
    text_query_service: TextQueryService = Depends(Factory().get_text_query_service),
    object_query_service: ObjectQueryService = Depends(
        Factory().get_object_query_service
    ),
):
    return QueryController(
        text_query_serivce=text_query_service, object_query_service=object_query_service
    )


@query_router.post(
    "/search",
    response_model=Response[List[KeyframeWithConfidence]],
    summary="Search keyframes",
    tags=["Query"],
    response_description="List of keyframe indexes",
    status_code=200,
    description="Predict to get top k similar keyframes",
)
async def search(
    text_query_service: TextQueryService = Depends(Factory().get_text_query_service),
    object_query_service: ObjectQueryService = Depends(
        Factory().get_object_query_service
    ),
    request_body: List[SearchBodyRequest] = Body(),
    vector_search: str = Query(example="faiss", description="Description for param1"),
    k_query: int = Query(5, description="kquery vector search"),
):
    settings = SearchSettings(vector_search=vector_search, k_query=k_query)

    query_controller = get_query_controller(text_query_service, object_query_service)

    query = await query_controller.search_keyframes(request_body, settings)

    return Response[List[KeyframeWithConfidence]](data=query)


@query_router.get(
    "/object/names",
    tags=["Object"],
    response_model=Response[List[str]],
    status_code=200,
    description="Get Object Names",
)
async def query_one_by_index(
    query_service: ObjectQueryService = Depends(Factory().get_object_query_service),
) -> Response[List[str]]:
    query_controller = ObjectController(object_query_service=query_service)

    query = await query_controller.get_object_names()

    return Response[List[str]](data=query)
