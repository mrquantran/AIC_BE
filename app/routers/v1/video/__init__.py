from fastapi import APIRouter
from .video import video_router

video_router_api = APIRouter()
video_router_api.include_router(video_router, prefix="/video", tags=["Video"])
