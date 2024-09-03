import os
import json
from typing import List, Tuple
import numpy as np
import torch
import open_clip
import faiss
from usearch.index import Index as UsearchIndex

from app.services.setencebert import SentenceBertEmbedding

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def normalize(vector):
    """
    Normalize the input vector to unit length.

    Parameters:
    vector (np.ndarray): Input vector or matrix. If 2D, normalize each row independently.

    Returns:
    np.ndarray: Normalized vector or matrix.
    """
    if vector.ndim == 1:
        return vector / np.linalg.norm(vector)
    elif vector.ndim == 2:
        return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]
    else:
        raise ValueError("Input must be a 1D or 2D numpy array")


# Factory for creating model and preprocessing
class ModelFactory:
    @staticmethod
    def create_model(model_name: str, device: str):
        try:
            print(f"Attempting to load model on {device}")
            model, _, preprocess = open_clip.create_model_and_transforms(model_name)
            return model.to(device), preprocess
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU out of memory. Falling back to CPU.")
                device = "cpu"
                model, _, preprocess = open_clip.create_model_and_transforms(model_name)
                return model.to(device), preprocess
            else:
                raise e


# Strategy pattern for different indexing methods
class IndexStrategy:
    def build_index(self, embeddings: np.ndarray):
        raise NotImplementedError

    def save_index(self, file_path: str):
        raise NotImplementedError

    def load_index(self, file_path: str):
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[int, float]]:
        raise NotImplementedError


class FaissIndexStrategy(IndexStrategy):
    def __init__(self):
        self.faiss_index = None

    def build_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)

    def save_index(self, file_path: str):
        faiss.write_index(self.faiss_index, file_path)
        print(f"FAISS index saved to {file_path}")

    def load_index(self, file_path: str):
        self.faiss_index = faiss.read_index(file_path)
        print(f"FAISS index loaded from {file_path}")

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[int, float]]:
        # Ensure query_embedding is 2D and has the correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.faiss_index.d:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[1]} does not match index dimension {self.faiss_index.d}"
            )

        faiss.normalize_L2(query_embedding)
        distances, indices = self.faiss_index.search(query_embedding, k)
        return list(zip(indices[0], distances[0]))

    def search_in_ranges(
        self, query_embedding: np.ndarray, ranges: List[Tuple[int, int]], k: int
    ) -> List[Tuple[int, float]]:
        # Flatten the range of tuples into a list of indices
        filter_ids = []
        for start, end in ranges:
            filter_ids.extend(range(start, end + 1))
        print(f"Searching in FAISS within specified ranges: {filter_ids}")

        # Create an ID selector for the specified ranges
        id_selector = faiss.IDSelectorArray(np.array(filter_ids, dtype=np.int64))

        # Prepare search parameters with the selector
        params = faiss.SearchParametersIVF()
        params.sel = id_selector

        # Ensure query_embedding is 2D and normalized
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.faiss_index.d:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[1]} does not match index dimension {self.faiss_index.d}"
            )

        faiss.normalize_L2(query_embedding)
        distances, indices = self.faiss_index.search(query_embedding, k, params=params)
        return list(zip(indices[0], distances[0]))


class UsearchIndexStrategy(IndexStrategy):
    def __init__(self):
        self.usearch_index = None

    def build_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        self.usearch_index = UsearchIndex(ndim=dimension, metric="cosine")
        for i, embedding in enumerate(embeddings):
            self.usearch_index.add(i, embedding)

    def save_index(self, file_path: str):
        self.usearch_index.save(file_path)
        print(f"USearch index saved to {file_path}")

    def load_index(self, file_path: str, dimension: int = 1024):
        self.usearch_index = UsearchIndex(ndim=dimension, metric="cosine")
        self.usearch_index.load(file_path)
        print(f"USearch index loaded from {file_path}")

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[int, float]]:
        print("Searching in Usearch")
        matches = self.usearch_index.search(query_embedding, k)
        return [(int(match.key), match.distance) for match in matches]

    def search_in_ranges(
        self, query_embedding: np.ndarray, ranges: List[Tuple[int, int]], k: int
    ) -> List[Tuple[int, float]]:
        """
        Searches for the top-k nearest embeddings within specified ranges.

        :param query_embedding: The query embedding vector.
        :param ranges: A list of tuples, each specifying a (min, max) range of IDs to consider in the search.
        :param k: The number of top results to return.
        :return: A list of tuples (index, distance) for the top-k nearest neighbors within the ranges.
        """
        print("Searching in Usearch within specified ranges")
        matches = self.usearch_index.search(
            query_embedding, k * 10
        )  # Perform a larger search to filter later
        filtered_matches = []

        # Filter matches based on the provided ranges
        for match in matches:
            key = int(match.key)
            for range_min, range_max in ranges:
                if range_min <= key <= range_max:
                    filtered_matches.append((key, match.distance))
                    break  # No need to check further ranges if match is found

        # Sort the filtered matches based on distance and take the top k results
        filtered_matches = sorted(filtered_matches, key=lambda x: x[1])[:k]

        return filtered_matches


class CLIPEmbedding:
    def __init__(self, model_name: str, model_nick_name: str, device: str = None):
        self.model_nick_name = model_nick_name
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.preprocess = ModelFactory.create_model(model_name, self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.faiss_strategy = FaissIndexStrategy()
        self.usearch_strategy = UsearchIndexStrategy()
        self.audio_usearch_strategy = UsearchIndexStrategy()
        self.global_index2image_path = {}

    async def text_query(
        self,
        query: str,
        k: int = 20,
        use_faiss: bool = True,
        ranges: List[Tuple[int, int]] = None,
    ) -> List[Tuple[int, float]]:
        with torch.no_grad():
            text_tokens = self.tokenizer([query]).to(self.device)
            query_embedding = (
                self.model.encode_text(text_tokens)
                .cpu()
                .detach()
                .numpy()
                .astype(np.float32)
            )

        # Get the dimension of the FAISS index
        if use_faiss:
            index_dimension = self.faiss_strategy.faiss_index.d
        else:
            index_dimension = self.usearch_strategy.usearch_index.ndim

        # Resize and normalize the query embedding if necessary
        if query_embedding.shape[1] != index_dimension:
            print(
                f"Resizing query embedding from {query_embedding.shape[1]} to {index_dimension}"
            )
            query_embedding = np.resize(query_embedding, (1, index_dimension))
            query_embedding = normalize(query_embedding)

        if use_faiss:
            if len(ranges) > 0:
                return self.faiss_strategy.search_in_ranges(query_embedding, ranges, k)
            else:
                return self.faiss_strategy.search(query_embedding, k)
        else:
            if len(ranges) > 0:
                return self.usearch_strategy.search_in_ranges(
                    query_embedding[0], ranges, k
                )
            else:
                return self.usearch_strategy.search(query_embedding[0], 500)

    def get_image_paths(self, indices: List[int]) -> List[str]:
        return [self.global_index2image_path.get(i, "Unknown") for i in indices]

        # Add an audio query function if needed

    async def audio_query_by_text(
        self,
        text_query: str,
        k: int = 20,
        ranges: List[Tuple[int, int]] = None,
    ) -> List[Tuple[int, float]]:
        audio_embedding = SentenceBertEmbedding().embed(text_query)
        audio_embedding = normalize(
            audio_embedding.reshape(1, -1)
        )  # Normalize embedding

        if ranges:
            return self.audio_usearch_strategy.search_in_ranges(
                audio_embedding, ranges, k
            )
        else:
            return self.audio_usearch_strategy.search(audio_embedding, k)

    def load_indexes(
        self,
        faiss_path: str = None,
        usearch_path: str = None,
        global2imgpath_path: str = None,
        audio_usearch_path: str = None,
    ):
        if faiss_path:
            self.faiss_strategy.load_index(faiss_path)

        if usearch_path:
            self.usearch_strategy.load_index(usearch_path, dimension=1024)

        if audio_usearch_path:
            self.audio_usearch_strategy.load_index(audio_usearch_path, dimension=768)

        if global2imgpath_path:
            with open(global2imgpath_path, "r") as f:
                self.global_index2image_path = json.load(f)
        print("Indexes and mappings loaded successfully.")
