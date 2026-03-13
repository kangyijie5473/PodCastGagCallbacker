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
            podcast_path = os.path.join(self.index_dir, podcast_name)
            if not os.path.isdir(podcast_path):
                continue
                
            # Walk 2: Audio Directories
            for audio_id in os.listdir(podcast_path):
                audio_path = os.path.join(podcast_path, audio_id)
                if os.path.isdir(audio_path) and os.path.exists(os.path.join(audio_path, "windows.json")):
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
            self.load_index(item['podcast'], item['audio_id'])
        
        # Filter valid keys that are actually loaded
        valid_keys = [f"{item['podcast']}/{item['audio_id']}" for item in target_indices]
        valid_keys = [k for k in valid_keys if k in self.indices]
        
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
                # Store metadata for frontend
                candidates.append({
                    "score": float(scores[idx]), # Vector score
                    "text": windows[idx]["text"],
                    "window_id": int(idx), # Convert numpy.int64 to python int
                    "metadata": {
                        "podcast": p_name,
                        "audio_id": a_id,
                        "start": float(windows[idx]["start"]), # Ensure float
                        "end": float(windows[idx]["end"]) # Ensure float
                    }
                })
        
        # Sort combined results by vector score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_results = candidates[:initial_k]
        
        # 2. Second Pass: Reranking (if enabled)
        if self.reranker:
            # Prepare docs for reranker
            texts = [res["text"] for res in top_results]
            if texts:
                # Use the 'rerank' method which takes query and list of docs
                rerank_scores = self.reranker.rerank(query, texts)
                
                # Update scores and resort
                for i, res in enumerate(top_results):
                    if i < len(rerank_scores):
                        res["score"] = float(rerank_scores[i])
            
            top_results.sort(key=lambda x: x["score"], reverse=True)
            
        return top_results[:top_k]
