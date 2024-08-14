from typing import Any, Generic, Type, TypeVar, List
from uuid import UUID
from pydantic import BaseModel
from pymongo.collection import Collection
from bson.objectid import ObjectId
from app.common.exceptions import NotFoundException
from app.common.repository.base import BaseRepository

ModelType = TypeVar("ModelType", bound=dict)

class BaseController(Generic[ModelType]):
    """Base class for data controllers."""

    def __init__(self, model: Type[ModelType], repository: BaseRepository[ModelType]):
        self.model_class = model
        self.repository = repository

    async def get_by_id(self, id_: str) -> ModelType:
        """
        Returns the model instance matching the id.

        :param id_: The id to match.
        :return: The model instance.
        """
        db_obj = await self.repository.get_by(field="_id", value=ObjectId(id_), unique=True)
        if not db_obj:
            raise NotFoundException(
                f"{self.model_class.__name__.title()} with id: {id_} does not exist")
        return db_obj

    async def get_by_uuid(self, uuid: UUID) -> ModelType:
        """
        Returns the model instance matching the uuid.

        :param uuid: The uuid to match.
        :return: The model instance.
        """
        db_obj = await self.repository.get_by(field="uuid", value=str(uuid), unique=True)
        if not db_obj:
            raise NotFoundException(
                f"{self.model_class.__name__.title()} with uuid: {uuid} does not exist")
        return db_obj

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """
        Returns a list of records based on pagination params.

        :param skip: The number of records to skip.
        :param limit: The number of records to return.
        :return: A list of records.
        """
        response = await self.repository.get_all(skip, limit)
        return response

    async def create(self, attributes: dict[str, Any]) -> ModelType:
        """
        Creates a new Object in the DB.

        :param attributes: The attributes to create the object with.
        :return: The created object.
        """
        create = await self.repository.create(attributes)
        return create

    async def delete(self, id_: str) -> bool:
        """
        Deletes the Object from the DB.

        :param id_: The id of the model to delete.
        :return: True if the object was deleted, False otherwise.
        """
        delete = await self.repository.delete(ObjectId(id_))
        return delete

    @staticmethod
    def extract_attributes_from_schema(schema: BaseModel, excludes: set = None) -> dict[str, Any]:
        """
        Extracts the attributes from the schema.

        :param schema: The schema to extract the attributes from.
        :param excludes: The attributes to exclude.
        :return: The attributes.
        """
        return schema.dict(exclude=excludes, exclude_unset=True)
