from typing import Any, Generic, Type, TypeVar, List, Optional

from pymongo.collection import Collection
from bson.objectid import ObjectId

ModelType = TypeVar("ModelType", bound=dict)

class BaseRepository(Generic[ModelType]):
    """Base class for data repositories."""

    def __init__(self, collection: Collection):
        self.collection = collection

    async def create(self, attributes: dict[str, Any] = None) -> ModelType:
        """
        Creates the model instance.

        :param attributes: The attributes to create the model with.
        :return: The created model instance.
        """
        if attributes is None:
            attributes = {}
        result = await self.collection.insert_one(attributes)
        return {**attributes, "_id": result.inserted_id}

    async def get_all(
        self, skip: int = 0, limit: int = 100
    ) -> List[ModelType]:
        """
        Returns a list of model instances.

        :param skip: The number of records to skip.
        :param limit: The number of record to return.
        :return: A list of model instances.
        """
        cursor = self.collection.find().skip(skip).limit(limit)
        return await cursor.to_list(length=limit)

    async def get_by(
        self, field: str, value: Any, unique: bool = False
    ) -> Optional[ModelType]:
        """
        Returns the model instance matching the field and value.

        :param field: The field to match.
        :param value: The value to match.
        :param unique: Whether to expect a unique result.
        :return: The model instance or None.
        """
        if unique:
            return await self.collection.find_one({field: value})
        else:
            cursor = self.collection.find({field: value})
            return await cursor.to_list(length=100)  # Default limit to 100

    async def delete(self, model_id: str) -> None:
        """
        Deletes the model.

        :param model_id: The ID of the model to delete.
        :return: None
        """
        await self.collection.delete_one({"_id": ObjectId(model_id)})

    async def _count(self) -> int:
        """
        Returns the count of the records.

        :return: The count of records.
        """
        return await self.collection.count_documents({})
    
    
