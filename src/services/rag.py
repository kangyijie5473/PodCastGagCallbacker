import os
import json
from typing import List, Dict, Any
from src.services.searcher import SearchService
from src.models.llm import LLMModel

class RAGService:
    def __init__(self, searcher: SearchService, llm: LLMModel):
        self.searcher = searcher
        self.llm = llm
        self.segments_cache = {} # audio_id -> list of segments

    def _get_segments(self, podcast_name: str, audio_id: str) -> List[Dict]:
        key = f"{podcast_name}/{audio_id}"
        if key in self.segments_cache:
            return self.segments_cache[key]
        
        # Path: data/{podcast_name}/{audio_id}/segments.json
        path = os.path.join(self.searcher.index_dir, podcast_name, audio_id, "segments.json")
        if not os.path.exists(path):
            return []
            
        with open(path, "r", encoding="utf-8") as f:
            segments = json.load(f)
            
        self.segments_cache[key] = segments
        return segments

    def _format_context(self, results: List[Dict]) -> str:
        context_parts = []
        for i, res in enumerate(results):
            podcast = res.get("podcast", "Unknown")
            audio_id = res["audio_id"]
            window = res["window"]
            score = res["score"]
            
            # Try to get detailed segments if available
            indices = window.get("segment_indices")
            formatted_content = ""
            
            if indices:
                start_idx, end_idx = indices
                all_segments = self._get_segments(podcast, audio_id)
                if all_segments:
                    chunk_segments = all_segments[start_idx:end_idx]
                    lines = []
                    for seg in chunk_segments:
                        spk = seg.get("speaker", "?")
                        text = seg.get("text", "")
                        start = seg.get("start", 0)
                        lines.append(f"[{start:.1f}s] Speaker {spk}: {text}")
                    formatted_content = "\n".join(lines)
            
            # Fallback to window text if segments loading failed
            if not formatted_content:
                formatted_content = window["text"]

            start_t = window.get("start", 0.0)
            end_t = window.get("end", 0.0)
            context_parts.append(f"Fragment {i+1} (Podcast: {podcast}, Episode: {audio_id}, Time: {start_t:.1f}-{end_t:.1f}s, Score: {score:.4f}):\n{formatted_content}\n")
            
        return "\n".join(context_parts)

    def answer(self, query: str, podcast_name: str = None, audio_id: str = None, top_k: int = 5, results: List[Dict] = None) -> str:
        # 1. Retrieve (if not provided)
        if results is None:
            results = self.searcher.search(query, podcast_name=podcast_name, audio_id=audio_id, top_k=top_k)
            
        if not results:
            return "No relevant podcast content found to answer your question."

        # 2. Format Context
        context_str = self._format_context(results)
        # 3. Construct Prompt
        system_prompt = (
            "You are a helpful podcast assistant. Your task is to answer the user's question "
            "based ONLY on the provided podcast transcripts. \n"
            "If the provided context does not contain the answer, say so.\n"
            "Cite the source Podcast, Episode and time range when possible.\n"
            "The transcripts include speaker IDs (e.g., Speaker 0, Speaker 1). Use this to distinguish who said what."
        )
        
        user_prompt = (
            f"User Question: {query}\n\n"
            f"Retrieved Context:\n"
            f"{context_str}\n\n"
            f"Please answer the question based on the context above."
        )
        
        # 4. Generate
        print("Generating answer with LLM...")
        answer = self.llm.generate(user_prompt, system_prompt=system_prompt)
        return answer
