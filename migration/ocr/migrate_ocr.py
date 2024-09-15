import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
import json
from typing import Dict, List, Union
import asyncio



from app.models.ocr import OCR
from app.config.config import Settings


settings = Settings()

print(
    f"Connecting to mongodb at {settings.MONGO_HOST}:{settings.MONGO_PORT}/{settings.MONGO_DB}"
)

print(f"Using {settings.MONGO_USER} as username")
print(f"Using {settings.MONGO_PASSWORD} as password")

async def init_db():
    client = AsyncIOMotorClient(
        settings.MONGO_HOST,
        settings.MONGO_PORT,
        # username=settings.MONGO_USER,
        # password=settings.MONGO_PASSWORD,
    )
    await init_beanie(database=client[settings.MONGO_DB], document_models=[OCR])


def load_json_data(file_path: str) -> Dict[str, str]:
    with open(file_path, "r") as f:
        print(f"Loading data from {file_path}")
        return json.load(f)


def transform_data(data: Dict[str, str]) -> List[Dict[str, Union[int, str]]]:
    # split value to get group_id, video_id, frame_id

    return [OCR(key=int(key), value=value,
                video_id=int(value.split("/")[0]),
                group_id=int(value.split("/")[1]),
                frame_id=int(value.split("/")[2])
            ) for key, value in data.items()]

path = os.path.join(os.path.dirname(__file__), "./output.json")


async def migrate(file_path: str = "./output.json"):
    await init_db()

    data = load_json_data(file_path)
    transformed_data = transform_data(data)
    print(f"[55] Loaded {len(transformed_data)} documents")

    await OCR.delete_all()
    print("Deleted all documents")

    if transformed_data:
        for doc in transformed_data:
            print(f"Inserted {doc.value} documents")
            await OCR.insert(doc)

    else:
        print("No data to insert")


if __name__ == "__main__":
    asyncio.run(migrate(path))
