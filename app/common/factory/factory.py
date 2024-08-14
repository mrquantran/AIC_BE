from app.models.object import Object
from app.repositories.object_query import ObjectQueryRepository
from app.repositories.text_query import TextQueryRepository
from app.models.text import Text
from app.services.object_query import ObjectQueryService
from app.services.text_query import TextQueryService


class Factory:
    """
    This is the factory container that will instantiate all the controllers and
    repositories which can be accessed by the rest of the application.
    """
    def text_query_repository(self):
        return TextQueryRepository(collection=Text)

    def get_text_query_service(self):
        return TextQueryService(query_repository=self.text_query_repository())

    def object_query_repository(self):
        return ObjectQueryRepository(collection=Object)

    def get_object_query_service(self):
        return ObjectQueryService(query_repository=self.object_query_repository())
