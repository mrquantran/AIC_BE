import os
from typing import List
from fastapi import FastAPI, Request
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.common.exceptions import CustomException
from app.routers.v1 import v1_router
from app.config import settings
from app.models.text import Text
from beanie import init_beanie
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from app.common.middlewares import ResponseLoggerMiddleware
from fastapi.staticfiles import StaticFiles

from app.services.clip_embedding import CLIPEmbedding

def init_listeners(app_: FastAPI) -> None:
    @app_.exception_handler(CustomException)
    async def custom_exception_handler(request: Request, exc: CustomException):
        return JSONResponse(
            status_code=exc.code,
            content={"error_code": exc.error_code, "message": exc.message},
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application", settings)
    # Setup mongoDB
    app.client = AsyncIOMotorClient(
        settings.MONGO_HOST,
        settings.MONGO_PORT,
        # username=settings.MONGO_USER,
        # password=settings.MONGO_PASSWORD,
    )
    await init_beanie(database=app.client[settings.MONGO_DB], document_models=[Text])

    yield


def make_middleware() -> List[Middleware]:
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
        Middleware(ResponseLoggerMiddleware),
    ]
    return middleware


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
    middleware=make_middleware(),
    docs_url= None if settings.ENVIRONMENT == "production" else "/docs",
)

init_listeners(app_=app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_path = os.path.join(os.path.dirname(__file__), "./data/images")
app.mount("/images", StaticFiles(directory=image_path, html=False), name="images")

app.include_router(v1_router, prefix=settings.API_V1_STR)
