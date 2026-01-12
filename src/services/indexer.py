import os
import json
import numpy as np
from typing import List, Dict, Any
from src.models.base import ASRModel, EmbeddingModel

class IndexingService:
    def __init__(self, asr_model: ASRModel, embed_model: EmbeddingModel, index_dir: str):
        self.asr = asr_model
        self.embed = embed_model
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)

    def process_audio(self, audio_path: str, audio_id: str, window_size: int = 20, stride: int = 5):
        # 1. Transcribe
        print(f"Transcribing {audio_path}...")
        segments = self.asr.transcribe(audio_path)
        
        # 2. Save raw segments
        audio_dir = os.path.join(self.index_dir, audio_id)
        os.makedirs(audio_dir, exist_ok=True)
        
        with open(os.path.join(audio_dir, "segments.json"), "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        # 3. Create Windows (Story chunks)
        print(f"Creating windows (size={window_size}, stride={stride})...")
        windows = self._create_windows(segments, window_size=window_size, stride=stride)
        
        # 4. Embed Windows
        print(f"Embedding {len(windows)} windows...")
        texts = [w["text"] for w in windows]
        if texts:
            embeddings = self.embed.encode(texts, is_query=False)
            
            # 5. Save Index
            np.save(os.path.join(audio_dir, "embeddings.npy"), embeddings)
            with open(os.path.join(audio_dir, "windows.json"), "w", encoding="utf-8") as f:
                json.dump(windows, f, ensure_ascii=False, indent=2)
        
        print(f"Indexing complete for {audio_id}")
        return {
            "id": audio_id,
            "segments_count": len(segments),
            "windows_count": len(windows)
        }

    def _create_windows(self, segments, window_size, stride):
        windows = []
        n = len(segments)
        for i in range(0, n, stride):
            end_idx = min(i + window_size, n)
            if i >= end_idx: break
            
            chunk = segments[i:end_idx]
            if not chunk: continue
            
            # Aggregate text for embedding (semantic search)
            text = " ".join([s["text"] for s in chunk])
            
            # Aggregate speaker info for display/filtering
            speakers = list(set([s.get("speaker", 0) for s in chunk]))
            
            start_time = chunk[0]["start"]
            end_time = chunk[-1]["end"]
            
            windows.append({
                "window_id": i,
                "start": start_time,
                "end": end_time,
                "text": text,
                "speakers": speakers,
                "segment_indices": [i, end_idx] # Range [start, end)
            })
            
            if end_idx == n:
                break
        return windows
