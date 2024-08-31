from app.models.object import Object
from app.models.ocr import OCR
from app.repositories.object_query import ObjectQueryRepository
from app.repositories.ocr_query import OCRQueryRepository
from app.repositories.text_query import TextQueryRepository
from app.models.keyframe import Keyframe
from app.services.object_query import ObjectQueryService
from app.services.ocr_query import OCRQueryService
from app.services.reciprocal_rank_fusion import ReciporalRankFusionService
from app.services.text_query import TextQueryService


class Factory:
    """
    This is the factory container that will instantiate all the controllers and
    repositories which can be accessed by the rest of the application.
    """

    def text_query_repository(self):
        return TextQueryRepository(collection=Keyframe)

    def get_text_query_service(self):
        return TextQueryService(query_repository=self.text_query_repository())

    def object_query_repository(self):
        return ObjectQueryRepository(collection=Object)

    def get_object_query_service(self):
        return ObjectQueryService(query_repository=self.object_query_repository())

    def get_ocr_query_service(self):
        return OCRQueryService(query_repository=self.ocr_query_repository())

    def ocr_query_repository(self):
        return OCRQueryRepository(collection=OCR)
    
    def get_reciprocal_rank_fusion_service(self):
        return ReciporalRankFusionService()
