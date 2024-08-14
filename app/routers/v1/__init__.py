from fastapi import APIRouter
from .health import monitoring_router
from .query import query_router_api

v1_router = APIRouter()
v1_router.include_router(monitoring_router)
v1_router.include_router(query_router_api)
