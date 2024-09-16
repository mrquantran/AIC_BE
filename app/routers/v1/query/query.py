from typing import List
from app.common.factory import Factory
from app.controllers.object import ObjectController
from app.schemas.requests.query import GetNearestIndexRequest, GetRangeIndexRequest
from app.schemas.responses.keyframes import KeyFrameInformation, KeyframeWithConfidence
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
    ocr_query_service=Depends(Factory().get_ocr_query_service),
):
    return QueryController(
        text_query_serivce=text_query_service,
        object_query_service=object_query_service,
        ocr_query_service=ocr_query_service,
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
    ocr_query_service=Depends(Factory().get_ocr_query_service),
    request_body: List[SearchBodyRequest] = Body(),
    vector_search: str = Query(example="faiss", description="Description for param1"),
    k_query: int = Query(5, description="kquery vector search"),
    display: int = Query(5, description="display vector search"),
    filter_indexes: List[int] = Query(
        None, description="List of indexes to filter the search"
    ),
):
    settings = SearchSettings(
        vector_search=vector_search, k_query=k_query, display=display
    )

    query_controller = get_query_controller(
        text_query_service, object_query_service, ocr_query_service
    )

    query = await query_controller.search_keyframes(request_body, settings, filter_indexes)

    return Response[List[KeyframeWithConfidence]](data=query)


@query_router.post(
    "/index/range",
    response_model=Response[List[KeyframeWithConfidence]],
    status_code=200,
    description="Get Index By Keyframe Information",
)
async def query_keyframe_by_range(
    request_body: List[GetRangeIndexRequest] = Body(),
    query_service: TextQueryService = Depends(Factory().get_text_query_service),
) -> Response[List[KeyframeWithConfidence]]:
    query = await query_service.get_keyframes_by_ranges(request_body)
    return Response[List[KeyframeWithConfidence]](data=query)


@query_router.post(
    "/index/nearest",
    response_model=Response[KeyFrameInformation],
    status_code=200,
    description="Get Index Nearby",
)
async def query_index_by_keframe_information(
    request_body: GetNearestIndexRequest = Body(),
    query_service: TextQueryService = Depends(Factory().get_text_query_service),
) -> Response[KeyFrameInformation]:
    group_id = request_body.group_id
    video_id = request_body.video_id
    keyframe_id = request_body.keyframe_id
    print(f"group_id: {group_id}, video_id: {video_id}, keyframe_id: {keyframe_id}")
    print(f"query_service: {query_service}")
    # print all method query_service
    query = await query_service.get_nearest_index(
        group_id= int(group_id), video_id= int(video_id), keyframe_id= int(keyframe_id)
    )

    return Response[KeyFrameInformation](data=query)


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
