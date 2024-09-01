import os
import aiofiles
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse


video_router = APIRouter()

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "../../../../data/videos")

# Increase block size for potentially fewer I/O operations
BLOCK_SIZE = 65536


async def ranged(
    file_path: str, start: int = 0, end: int = None, block_size: int = BLOCK_SIZE
):
    async with aiofiles.open(file_path, mode="rb") as file:
        await file.seek(start)
        while True:
            read_size = min(block_size, end - start + 1) if end else block_size
            chunk = await file.read(read_size)
            if not chunk:
                break
            yield chunk


@video_router.get("/{group_id}/{video_id}")
# @cache(expire=3600)  # Cache for 1 hour
async def get_video(group_id: str, video_id: str, request: Request):
    video_path = os.path.join(
        VIDEO_PATH,
        f"Videos_L{group_id.zfill(2)}/video/L{group_id.zfill(2)}_V{video_id.zfill(3)}.mp4",
    )

    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = os.stat(video_path).st_size
    range_header = request.headers.get("Range")

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": "video/mp4",
    }

    start = 0
    end = file_size - 1
    status_code = 200

    if range_header:
        start, end = range_header.replace("bytes=", "").split("-")
        start = int(start) if start else 0
        end = int(end) if end else file_size - 1
        if start >= file_size or end >= file_size:
            raise HTTPException(
                status_code=416, detail="Requested Range Not Satisfiable"
            )
        size = end - start + 1
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        headers["Content-Length"] = str(size)
        status_code = 206
    else:
        headers["Content-Length"] = str(file_size)

    return StreamingResponse(
        ranged(video_path, start=start, end=end),
        headers=headers,
        status_code=status_code,
    )


# # Initialize cache
# @video_router.on_event("startup")
# async def startup():
#     redis = aioredis.from_url(
#         "redis://localhost", encoding="utf8", decode_responses=True
#     )
#     FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
