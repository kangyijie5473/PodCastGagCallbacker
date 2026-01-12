import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from src.models.base import EmbeddingModel

class SearchService:
    def __init__(self, embed_model: EmbeddingModel, index_dir: str):
        self.embed = embed_model
        self.index_dir = index_dir
        self.indices = {} # Cache indices in memory for demo

    def load_index(self, audio_id: str):
        path = os.path.join(self.index_dir, audio_id)
        if not os.path.exists(path):
            return False
        
        with open(os.path.join(path, "windows.json"), "r", encoding="utf-8") as f:
            windows = json.load(f)
        
        embs = np.load(os.path.join(path, "embeddings.npy"))
        
        self.indices[audio_id] = {
            "windows": windows,
            "embeddings": embs
        }
        return True

    def list_indices(self):
        if not os.path.exists(self.index_dir):
            return []
        return [d for d in os.listdir(self.index_dir) if os.path.isdir(os.path.join(self.index_dir, d))]

    def search(self, query: str, audio_id: Optional[str] = None, top_k: int = 5):
        # If audio_id is provided, search only that. Else search all loaded (or load all available).
        
        target_ids = []
        if audio_id:
            target_ids = [audio_id]
        else:
            # For demo, load all available indices if not specified
            available = self.list_indices()
            target_ids = available

        # Ensure loaded
        for aid in target_ids:
            if aid not in self.indices:
                self.load_index(aid)
        
        valid_targets = [aid for aid in target_ids if aid in self.indices]
        if not valid_targets:
            return []

        q_emb = self.embed.encode([query], is_query=True)[0]
        
        results = []
        for aid in valid_targets:
            data = self.indices[aid]
            windows = data["windows"]
            embs = data["embeddings"]
            
            # Cosine similarity
            scores = np.dot(embs, q_emb)
            
            # Get top K indices
            # If K is larger than size, take all
            k = min(top_k, len(scores))
            top_indices = np.argsort(scores)[::-1][:k]
            
            for idx in top_indices:
                results.append({
                    "score": float(scores[idx]),
                    "audio_id": aid,
                    "window": windows[idx]
                })
        
        # Sort combined results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
