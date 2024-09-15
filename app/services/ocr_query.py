import asyncio
from typing import List, Set, Tuple, Union
# import asyncio
# from app.common.controller import BaseController
# from app.common.enum import QueryType
# from app.models.ocr import OCR
# from app.repositories.ocr_query import OCRQueryRepository
# from app.schemas.requests.query import SearchSettings
# from app.schemas.responses.keyframes import KeyframeWithConfidence
# from app.config.embedding import embedder
import unicodedata
from app.common.controller.base import BaseController
from app.config.ocr import OCR_CONTENT
from app.models.ocr import OCR
from app.repositories.ocr_query import OCRQueryRepository
from app.schemas.responses.keyframes import KeyframeWithConfidence
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np

class OCRQueryService(BaseController[OCR]):

    def __init__(self, query_repository: OCRQueryRepository):
        super().__init__(model=OCR, repository=query_repository)
        self.query_repository = query_repository
        self.ocr_content = OCR_CONTENT
    

    # ocr_queries is list tuple type [('USD', 0.6666666666666665, 8450)]  OCR content, confidence score, video ID
    async def search_keyframes_by_ocr(self, ocr_queries: List[Tuple[str, float, int]]) -> List[KeyframeWithConfidence]:
        # search in query_repository
        # return list of KeyframeWithConfidence
        keyframe_ocr = await self.query_repository.get_keyframe_by_indices([ocr_query[2] for ocr_query in ocr_queries])
        flattened_results = {
            int(idx): score
            for (text, score, idx) in ocr_queries
        }

        keyframes_with_confidence = [
            KeyframeWithConfidence(
                key=keyframe.key,
                value=keyframe.value,
                confidence=flattened_results[keyframe.key],
                video_id=keyframe.video_id,
                group_id=keyframe.group_id,
            )
            for keyframe in keyframe_ocr
        ]

        return keyframes_with_confidence

        

    def normalize_text(self, text: str) -> str:
        """Normalize text by removing diacritics, special characters, converting lowercase, normalizing whitespace
        """

        text = ''.join(
            char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn'
        )
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def fuzzy_search(
        self,
        query: str, 
        top_k: int = 5,
        threshold: float = 0.5,
        stopwords: Union[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """ Performs a fuzzy search over a list of texts and returns texts sorted by similarity.

        Args:
            query (str): The query text
            text_list (List[str]): The list of texts to compare against
            top_k (int, optional): Number of top results to compare against. Defaults to 5.
            threshold (float, optional): Minimum similarity score threshold (0 to 1). . Defaults to 0.5.
            stopwords (Union[Set[str]], optional): Set of stopwords to remove . Defaults to None.

        Returns:
            List[Tuple[str, float]]: List of tuples with text and their similarity scores
        """ 
        text_list = self.ocr_content
        normalized_text_list = [self.normalize_text(text) for text in text_list]
        normalized_query = self.normalize_text(query)

        # if stopwords:
        #     normalized_texts = [remove_stopwords(text, stopwords) for text in normalized_texts]
        #     normalized_query = remove_stopwords(normalized_query, stopwords)

        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,5))
        tfidf_matrix = vectorizer.fit_transform(normalized_text_list)
        query_vector = vectorizer.transform([normalized_query])
        print(tfidf_matrix.shape, query_vector.shape)

        cosine_similarities = tfidf_matrix.dot(query_vector.T).toarray().ravel()

        indices = np.where(cosine_similarities >= threshold)[0]

        if len(indices) == 0:
        # Fallback to RapidFuzz if no results above threshold
            scores = [
            fuzz.token_set_ratio(normalized_query, text) / 100.0
            for text in normalized_text_list
            ]
            results = sorted(
            zip(text_list, scores, range(len(text_list))),
            key=lambda x: x[1],
            reverse=True
            )[:top_k]
        else:
            # Get texts and scores above threshold
            results = [
                (text_list[i], cosine_similarities[i], i)
                for i in indices
            ]
            # Sort results by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]

        return results
