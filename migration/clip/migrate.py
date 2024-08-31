import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
import json
from typing import Dict, List, Union
import asyncio

from app.config.config import Settings
from app.models.keyframe import Keyframe


# Add the parent directory to the sys.path
settings = Settings()

print(
    f"Connecting to mongodb at {settings.MONGO_HOST}:{settings.MONGO_PORT}/{settings.MONGO_DB}")


async def init_db():
    client = AsyncIOMotorClient(
        settings.MONGO_HOST,
        settings.MONGO_PORT,
        # username=settings.MONGO_USER,
        # password=settings.MONGO_PASSWORD,
    )
    await init_beanie(database=client[settings.MONGO_DB], document_models=[Keyframe])


def load_json_data(file_path: str) -> Dict[str, str]:
    with open(file_path, 'r') as f:
        print(f"Loading data from {file_path}")
        return json.load(f)


def transform_data(data: Dict[str, str]) -> List[Keyframe]:
    keyframes = []
    for key, value in data.items():
        try:
            # Assuming the value format is "{group}/{video}/{frame}"
            group, video, frame = map(int, value.split("/"))

            keyframes.append(
                Keyframe(
                    key=key,  # Assign the frame to the 'key'
                    value=value,  # Keep the original value
                    group_id=group,  # Assign the group
                    video_id=video,  # Assign the video
                    frame_id=frame  # Assign the frame
                )
            )
        except ValueError as e:
            print(f"Skipping invalid entry {value}: {e}")
            continue

    return keyframes


path = os.path.join(os.path.dirname(__file__), 'global2imgpath.json')


async def migrate(file_path: str = 'global2imgpath.json'):
    await init_db()

    data = load_json_data(file_path)
    transformed_data = transform_data(data)
    print(f"[55] Loaded {len(transformed_data)} documents")

    await Keyframe.delete_all()
    print("Deleted all documents")

    if transformed_data:
        for doc in transformed_data:
            print(f"Inserted {doc.value} documents")
            await Keyframe.insert(doc)

    else:
        print("No data to insert")

if __name__ == '__main__':
    asyncio.run(migrate(path))
