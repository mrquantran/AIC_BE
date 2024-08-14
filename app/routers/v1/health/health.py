from fastapi import APIRouter

from app.schemas.extras.health import Health
from app.config.config import settings

health_router = APIRouter()

@health_router.get("/")
async def health() -> Health:
    return Health(version=settings.VERSION, status="Healthy")
