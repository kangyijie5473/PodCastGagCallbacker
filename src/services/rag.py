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
        # Path: data/{podcast_name}/{audio_id}/segments.json
        # Handle cases where index_dir might be missing
        if not self.searcher.index_dir:
            return []
        base_dir = os.path.join(self.searcher.index_dir, podcast_name, audio_id)
        candidates = [
            os.path.join(base_dir, "llm_segements.json"),
            os.path.join(base_dir, "segments.json")
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    segments = json.load(f)
                if isinstance(segments, list):
                    return segments
            except Exception:
                continue
        return []

    def _format_context(self, results: List[Dict]) -> str:
        context_parts = []
        for i, res in enumerate(results):
            # Extract metadata safely
            meta = res.get("metadata", {})
            podcast = meta.get("podcast", "Unknown")
            audio_id = meta.get("audio_id", "Unknown")
            start_t = meta.get("start", 0.0)
            end_t = meta.get("end", 0.0)
            score = res.get("score", 0.0)
            
            # Text content
            formatted_content = res.get("text", "")
            
            # Try to load detailed segments if window_id is available
            # We need to know which window index this corresponds to in windows.json
            # The searcher now returns 'window_id' which is the index in windows.json
            window_idx = res.get("window_id")
            
            if window_idx is not None:
                # Load windows.json to get segment indices
                try:
                    windows_path = os.path.join(self.searcher.index_dir, podcast, audio_id, "windows.json")
                    if os.path.exists(windows_path):
                        with open(windows_path, "r", encoding="utf-8") as f:
                            windows = json.load(f)
                        
                        if 0 <= window_idx < len(windows):
                            window = windows[window_idx]
                            rich_text = window.get("text_with_speakers")
                            if isinstance(rich_text, str) and rich_text.strip():
                                formatted_content = rich_text
                            indices = window.get("segment_indices")
                            
                            if indices and not formatted_content:
                                start_idx, end_idx = indices
                                all_segments = self._get_segments(podcast, audio_id)
                                if all_segments:
                                    chunk_segments = all_segments[start_idx:end_idx]
                                    lines = []
                                    for seg in chunk_segments:
                                        spk = seg.get("speaker", "?")
                                        text = seg.get("text", "")
                                        # start = seg.get("start", 0)
                                        lines.append(f"Speaker {spk}: {text}")
                                    formatted_content = "\n".join(lines)
                except Exception as e:
                    print(f"Error loading details for context: {e}")

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

    def answer_stream(self, query: str, podcast_name: str = None, audio_id: str = None, top_k: int = 5, results: List[Dict] = None):
        if results is None:
            results = self.searcher.search(query, podcast_name=podcast_name, audio_id=audio_id, top_k=top_k)

        if not results:
            yield "No relevant podcast content found to answer your question."
            return

        context_str = self._format_context(results)
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

        print("Generating streaming answer with LLM...")
        for token in self.llm.generate_stream(user_prompt, system_prompt=system_prompt):
            yield token
