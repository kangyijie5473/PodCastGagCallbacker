from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class ASRModel(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe audio file to text segments.
        Returns: List of dicts with keys: 'start', 'end', 'text', 'speaker'(optional)
        """
        pass

class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode list of texts to embeddings.
        is_query: if True, may add specific prefixes (e.g. for E5 models)
        Returns: Numpy array of vectors.
        """
        pass
