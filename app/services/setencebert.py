import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceBertEmbedding:
    """
    class for sentence embedding using SentenceBERT
    """

    # Sử dụng sentenceBERT để encode
    def __init__(
        self, model_name: str = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    ):
        self.model = SentenceTransformer(model_name)

    def embed(self, sentence: str) -> np.ndarray:
        return (
            self.model.encode(sentence, convert_to_tensor=True).cpu().detach().numpy()
        )
