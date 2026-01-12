from .base import EmbeddingModel
from typing import List
from sentence_transformers import SentenceTransformer  
import numpy as np

import os

class LocalEmbedding(EmbeddingModel):
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        # Check if local model exists
        local_path = os.path.join(os.getcwd(), "models_cache", "bge-small-zh-v1.5")
        if os.path.exists(local_path):
            print(f"Loading embedding model from local path: {local_path}")
            self.model_name = local_path
        else:
            self.model_name = model_name
            
        self.model = SentenceTransformer(self.model_name)
    
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        # Check against model name/path
        is_bge = "bge" in self.model_name.lower()
        
        if is_bge and is_query:
            # BGE instruction for query
            instruction = "为这个句子生成表示以用于检索相关文章："
            texts = [instruction + t for t in texts]
        elif "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            texts = [
                (prefix + t) if not t.startswith(prefix) else t 
                for t in texts
            ]
        
        return self.model.encode(texts, normalize_embeddings=True)
