import os
import json
import time
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
        self.enable_timing = os.getenv("SEARCH_TIMING_LOG", "1") == "1"

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
        total_start = time.perf_counter()
        stage_ms: Dict[str, float] = {}
        per_index_ms: List[tuple[str, float]] = []
        
        target_indices = []
        t0 = time.perf_counter()
        available = self.list_indices()
        stage_ms["list_indices"] = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        for item in available:
            if podcast_name and item["podcast"] != podcast_name:
                continue
            if audio_id and item["audio_id"] != audio_id:
                continue
            target_indices.append(item)
        stage_ms["filter_targets"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for item in target_indices:
            self.load_index(item['podcast'], item['audio_id'])
        stage_ms["load_index"] = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        valid_keys = [f"{item['podcast']}/{item['audio_id']}" for item in target_indices]
        valid_keys = [k for k in valid_keys if k in self.indices]
        stage_ms["build_valid_keys"] = (time.perf_counter() - t0) * 1000
        
        if not valid_keys:
            stage_ms["total"] = (time.perf_counter() - total_start) * 1000
            if self.enable_timing:
                print(f"[SearchTiming] {stage_ms} per_index_top=[]")
            return []

        t0 = time.perf_counter()
        q_emb = self.embed.encode([query], is_query=True)[0]
        stage_ms["embed_query"] = (time.perf_counter() - t0) * 1000
        
        initial_k = top_k * 10 if self.reranker else top_k
        
        candidates = []
        t0 = time.perf_counter()
        for key in valid_keys:
            idx_start = time.perf_counter()
            data = self.indices[key]
            windows = data["windows"]
            embs = data["embeddings"]
            p_name = data["podcast"]
            a_id = data["audio_id"]
            
            if embs.shape[1] != q_emb.shape[0]:
                print(f"Warning: Dimension mismatch for {p_name}/{a_id}. Index: {embs.shape[1]}, Query: {q_emb.shape[0]}. Skipping.")
                per_index_ms.append((key, (time.perf_counter() - idx_start) * 1000))
                continue

            scores = np.dot(embs, q_emb)
            
            k = min(initial_k, len(scores))
            top_indices = np.argsort(scores)[::-1][:k]
            
            for idx in top_indices:
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
            per_index_ms.append((key, (time.perf_counter() - idx_start) * 1000))
        stage_ms["vector_scan"] = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_results = candidates[:initial_k]
        stage_ms["sort_candidates"] = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        if self.reranker:
            texts = [res["text"] for res in top_results]
            if texts:
                rerank_scores = self.reranker.rerank(query, texts)
                
                for i, res in enumerate(top_results):
                    if i < len(rerank_scores):
                        res["score"] = float(rerank_scores[i])
            
            top_results.sort(key=lambda x: x["score"], reverse=True)
        stage_ms["rerank"] = (time.perf_counter() - t0) * 1000
        stage_ms["total"] = (time.perf_counter() - total_start) * 1000
        if self.enable_timing:
            per_index_top = sorted(per_index_ms, key=lambda x: x[1], reverse=True)[:5]
            print(f"[SearchTiming] {stage_ms} per_index_top={per_index_top}")
            
        return top_results[:top_k]
