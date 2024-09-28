"""Migrate object table."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
import asyncio
from typing import Dict, List
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from app.config.config import Settings
from app.models import Keyframe, Object


settings = Settings()

print(
    f"Connecting to mongodb at {settings.MONGO_HOST}:{settings.MONGO_PORT}/{settings.MONGO_DB}"
)

async def init_db():
    client = AsyncIOMotorClient(
        settings.MONGO_HOST,
        settings.MONGO_PORT,
    )
    await init_beanie(
        database=client[settings.MONGO_DB], document_models=[Keyframe, Object]
    )

def load_json_data(file_path: str) -> Dict[str, List[str]]:
    with open(file_path, "r") as f:
        print(f"Loading data from {file_path}")
        return json.load(f)

async def create_path_to_key_mapping() -> Dict[str, int]:
    texts = await Keyframe.find_all().to_list()
    return {text.value: text.key for text in texts}

async def transform_and_save_data(input_file: str, output_file: str):
    data = load_json_data(input_file)
    path_to_key_mapping = await create_path_to_key_mapping()
    print(path_to_key_mapping)
    transformed_data = {}
    for obj, paths in data.items():
        for path in paths:
            if path not in path_to_key_mapping:
                print(f"Path {path} not found in Text table")
        transformed_data[obj] = [
            (
                str(path_to_key_mapping.get(path))
                if path in path_to_key_mapping
                else None
            )
            for path in paths
        ]

        print(f"Transformed {len(paths)} paths for object {obj}")

    with open(output_file, "w") as f:
        json.dump(transformed_data, f, indent=2)

    print(f"Transformed data saved to {output_file}")

async def migrate_objects(file_path: str):
    data = load_json_data(file_path)

    # Xóa tất cả documents hiện có trong bảng Object
    await Object.delete_all()
    print("Deleted all existing objects")

    # Tạo và chèn các Object mới, loại bỏ các giá trị None
    objects_to_insert = [
        Object(name=name, value=[int(value) for value in values if value is not None])
        for name, values in data.items()
    ]
    await Object.insert_many(objects_to_insert)
    print(f"Inserted {len(objects_to_insert)} new objects")

async def migrate_tag_object_probability(filename: str):
    data = load_json_data(filename)
    for keyframe_index, info in data.items():
        keyframe_index = int(keyframe_index)
        print(f"Processing keyframe {keyframe_index}")
        # Tìm kiếm keyframe theo keyframe_index
        keyframe = await Keyframe.find_one(Keyframe.key == keyframe_index)

        if not keyframe:
            # raise ValueError(f"Keyframe with index {keyframe_index} does not exist.")
            print(f"Keyframe with index {keyframe_index} does not exist.")
            continue 

        # Cập nhật tags vào keyframe
        keyframe.tags.update({tag: prob for tag, prob in info["tag_prob"]})
        await keyframe.save()


async def main():
    input_file = os.path.join(
        os.path.dirname(__file__), "merged_inference_results_numbered_output.json"
    )
    object_files = os.path.join(os.path.dirname(__file__), "tag_to_indices.json")
    await init_db()

    # Di chuyển dữ liệu đã chuyển đổi vào bảng Object
    await migrate_objects(object_files)
    await migrate_tag_object_probability(input_file)


if __name__ == "__main__":
    asyncio.run(main())
