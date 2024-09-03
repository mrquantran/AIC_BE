import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
import asyncio
from typing import Dict, List
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from app.config.config import Settings
from app.models import Keyframe

settings = Settings()

print(
    f"Connecting to MongoDB at {settings.MONGO_HOST}:{settings.MONGO_PORT}/{settings.MONGO_DB}"
)


async def init_db():
    client = AsyncIOMotorClient(
        settings.MONGO_HOST,
        settings.MONGO_PORT,
    )
    await init_beanie(database=client[settings.MONGO_DB], document_models=[Keyframe])


def load_json_data(file_path: str) -> Dict[str, List[str]]:
    """
    Load JSON data from a file and return it as a dictionary.
    """
    with open(file_path, "r") as f:
        print(f"Loading data from {file_path}")
        return json.load(f)


async def migrate_audio_index(data: Dict[str, List[str]]):
    """
    Migrate audio_index into Keyframe model based on video_index.

    Args:
        data: A dictionary where the key is audio_index and the value is a list of video_index.
    """
    for audio_index, video_indices in data.items():
        audio_index = int(audio_index)  # Convert audio_index to integer
        for video_index in video_indices:
            video_index = int(video_index)  # Convert video_index to integer

            # Find the Keyframe by the video_index (key field in Keyframe model)
            keyframe = await Keyframe.find_one(Keyframe.key == video_index)
            print(f"Processing Keyframe with key {video_index} by {audio_index}")
            if not keyframe:
                print(f"Keyframe with key {video_index} does not exist.")
                continue

            # Update the audio_index field in the Keyframe
            keyframe.audio_index = audio_index
            await keyframe.save()  # Save the updated Keyframe
            print(
                f"Updated Keyframe with key {video_index} to audio_index {audio_index}"
            )


async def main():
    """
    Main function to run the migration script.
    """
    # Path to the input JSON file containing audio to video index mapping
    input_file = os.path.join(os.path.dirname(__file__), "global_audio_index.json")

    await init_db()  # Initialize the database connection

    # Load data from the JSON file
    data = load_json_data(input_file)

    # Migrate audio_index into Keyframe model based on the mapping data
    await migrate_audio_index(data)


if __name__ == "__main__":
    asyncio.run(main())
