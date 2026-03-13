import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from src.models.base import EmbeddingModel
try:
    from src.models.reranker import RerankerModel
except ImportError:
    RerankerModel = None

class SearchService:
    def __init__(self, embed_model: EmbeddingModel, index_dir: str, reranker: Optional[RerankerModel] = None):
        self.embed = embed_model
        self.index_dir = index_dir
        self.reranker = reranker
        self.indices = {} # Cache indices in memory for demo

    def load_index(self, podcast_name: str, audio_id: str):
        key = f"{podcast_name}/{audio_id}"
        path = os.path.join(self.index_dir, podcast_name, audio_id)
        if not os.path.exists(path):
            return False
        
        with open(os.path.join(path, "windows.json"), "r", encoding="utf-8") as f:
            windows = json.load(f)
        
        embs = np.load(os.path.join(path, "embeddings.npy"))
        
        self.indices[key] = {
            "windows": windows,
            "embeddings": embs,
            "podcast": podcast_name,
            "audio_id": audio_id
        }
        return True

    def list_indices(self) -> List[Dict[str, str]]:
        # Returns list of {"podcast": ..., "audio_id": ...}
        results = []
        if not os.path.exists(self.index_dir):
            return []
            
        # Walk 1: Podcast Directories
        for podcast_name in os.listdir(self.index_dir):
            podcast_dir = os.path.join(self.index_dir, podcast_name)
            if not os.path.isdir(podcast_dir):
                continue
                
            # Walk 2: Audio Directories
            for audio_id in os.listdir(podcast_dir):
                audio_dir = os.path.join(podcast_dir, audio_id)
                if os.path.isdir(audio_dir) and os.path.exists(os.path.join(audio_dir, "windows.json")):
                    results.append({"podcast": podcast_name, "audio_id": audio_id})
        return results

    def search(self, query: str, podcast_name: Optional[str] = None, audio_id: Optional[str] = None, top_k: int = 5):
        # Target specific podcast or specific audio, or all
        
        target_indices = []
        available = self.list_indices()
        
        for item in available:
            # Filter by podcast if specified
            if podcast_name and item["podcast"] != podcast_name:
                continue
            # Filter by audio_id if specified (must also match podcast if provided, or global uniqueness not guaranteed but here we assume podcast+audio_id is unique)
            if audio_id and item["audio_id"] != audio_id:
                continue
            target_indices.append(item)

        # Ensure loaded
        for item in target_indices:
            key = f"{item['podcast']}/{item['audio_id']}"
            if key not in self.indices:
                self.load_index(item['podcast'], item['audio_id'])
        
        valid_keys = [f"{item['podcast']}/{item['audio_id']}" for item in target_indices if f"{item['podcast']}/{item['audio_id']}" in self.indices]
        
        if not valid_keys:
            return []

        q_emb = self.embed.encode([query], is_query=True)[0]
        
        # 1. First Pass: Vector Search (High Recall)
        # If reranker is enabled, retrieve more candidates (e.g., top_k * 10)
        initial_k = top_k * 10 if self.reranker else top_k
        
        candidates = []
        for key in valid_keys:
            data = self.indices[key]
            windows = data["windows"]
            embs = data["embeddings"]
            p_name = data["podcast"]
            a_id = data["audio_id"]
            
            # Check dimension match
            if embs.shape[1] != q_emb.shape[0]:
                print(f"Warning: Dimension mismatch for {p_name}/{a_id}. Index: {embs.shape[1]}, Query: {q_emb.shape[0]}. Skipping.")
                continue

            # Cosine similarity
            scores = np.dot(embs, q_emb)
            
            # Get top K indices for this file
            # We can be generous here since we will sort globally later
            k = min(initial_k, len(scores))
            top_indices = np.argsort(scores)[::-1][:k]
            
            for idx in top_indices:
                candidates.append({
                    "score": float(scores[idx]), # Vector score
                    "podcast": p_name,
                    "audio_id": a_id,
                    "text": windows[idx]["text"],
                    "start": windows[idx]["start"],
                    "end": windows[idx]["end"],
                    "window": windows[idx]
                })
        
        # Sort combined results by vector score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Keep only top initial_k candidates globally
        candidates = candidates[:initial_k]
        
        # 2. Second Pass: Re-ranking (High Precision)
        if self.reranker:
            texts = [c["text"] for c in candidates]
            if texts:
                rerank_scores = self.reranker.rerank(query, texts)
                for i, score in enumerate(rerank_scores):
                    candidates[i]["vector_score"] = candidates[i]["score"] # Save vector score
                    candidates[i]["score"] = score # Overwrite with reranker score
                
                # Re-sort by reranker score
                candidates.sort(key=lambda x: x["score"], reverse=True)
                
        return candidates[:top_k]
