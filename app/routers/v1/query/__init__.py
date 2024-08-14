from fastapi import APIRouter
from .query import query_router

query_router_api = APIRouter()
query_router_api.include_router(query_router, prefix="/query", tags=["Query"])
