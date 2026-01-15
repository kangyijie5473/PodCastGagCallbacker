import os
import json
import numpy as np
import librosa
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from src.models.base import ASRModel, EmbeddingModel
from src.models.llm import LLMModel

class IndexingService:
    def __init__(self, asr_model: ASRModel, embed_model: EmbeddingModel, index_dir: str, llm: Optional[LLMModel] = None):
        self.asr = asr_model
        self.embed = embed_model
        self.index_dir = index_dir
        self.llm = llm
        os.makedirs(index_dir, exist_ok=True)

    def process_audio(self, audio_path: str, audio_id: str, podcast_name: str = "default", window_size: int = 20, stride: int = 5, progress_callback=None):
        # Helper to report progress safely
        def report(progress, msg):
            if progress_callback:
                progress_callback(progress, msg)
            tqdm.write(f"[{progress}%] {msg}")

        # 1. Transcribe
        report(5, f"Transcribing {audio_path}...")
        
        try:
            duration = librosa.get_duration(path=audio_path)
            # Estimate RTF (Real Time Factor) ~ 0.2 for ASR + Diarization
            estimated_seconds = duration * 0.2
            report(10, f"Audio Duration: {duration/60:.2f} min. Estimated time: {estimated_seconds/60:.2f} min (RTF ~0.2)")
        except Exception as e:
            tqdm.write(f"Could not estimate duration: {e}")

        segments = self.asr.transcribe(audio_path)
        report(60, "Transcription complete. Processing segments...")
        
        # 2. Save raw segments
        # Path: data/{podcast_name}/{audio_id}/
        audio_dir = os.path.join(self.index_dir, podcast_name, audio_id)
        os.makedirs(audio_dir, exist_ok=True)
        
        with open(os.path.join(audio_dir, "segments.json"), "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        # 3. Create Windows (Story chunks)
        report(70, f"Creating windows (size={window_size}, stride={stride})...")
        windows = self._create_windows(segments, window_size=window_size, stride=stride)
        
        # 3.1 Refine Windows with LLM (if enabled)
        if self.llm:
            windows = self._refine_windows(windows, progress_callback=lambda p, m: report(70 + int(p * 0.2), m))

        # 4. Embed Windows
        report(90, f"Embedding {len(windows)} windows...")
        texts = [w["text"] for w in windows]
        if texts:
            embeddings = self.embed.encode(texts, is_query=False)
            
            # 5. Save Index
            np.save(os.path.join(audio_dir, "embeddings.npy"), embeddings)
            with open(os.path.join(audio_dir, "windows.json"), "w", encoding="utf-8") as f:
                json.dump(windows, f, ensure_ascii=False, indent=2)
        
        report(100, f"Indexing complete for {audio_id}")
        return {
            "id": audio_id,
            "podcast": podcast_name,
            "segments_count": len(segments),
            "windows_count": len(windows)
        }

    def _refine_windows(self, windows: List[Dict], progress_callback=None) -> List[Dict]:
        tqdm.write(f"Refining {len(windows)} windows with LLM (this may take a while)...")
        
        refined_windows = []
        total = len(windows)
        for i, w in enumerate(tqdm(windows, desc="LLM Refinement", unit="win")):
            if progress_callback and i % 5 == 0: # Update every 5 items
                 progress_callback(i / total * 100, f"Refining window {i+1}/{total}")

            raw_text = w["text"]
            # Skip empty or very short text
            if len(raw_text) < 10:
                refined_windows.append(w)
                continue

            prompt = (
                "You are an expert editor. Refine the following ASR transcript:\n"
                "1. Fix punctuation, capitalization, and typos.\n"
                "2. Remove fillers (um, uh) and repetitions.\n"
                "3. Fix homophones based on context (e.g., 'notebook IOM' -> 'NotebookLM').\n"
                "4. Keep the original meaning and language (Chinese).\n"
                "5. Output ONLY the refined text.\n\n"
                f"Raw text:\n{raw_text}"
            )
            
            try:
                refined_text = self.llm.generate(prompt)
                # Basic check
                if refined_text and len(refined_text) > 5 and "Error" not in refined_text:
                    w["raw_text"] = raw_text
                    w["text"] = refined_text.strip()
            except Exception as e:
                tqdm.write(f"Refinement failed for window {w['window_id']}: {e}")
            
            refined_windows.append(w)
            
        return refined_windows

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
        return windows
