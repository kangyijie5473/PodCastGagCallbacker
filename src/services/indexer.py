import os
import json
import re
import numpy as np
import librosa
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from src.models.base import ASRModel, EmbeddingModel
from src.models.llm import LLMModel

class IndexingService:
    def __init__(self, asr_model: ASRModel, embed_model: EmbeddingModel, index_dir: str, llm: Optional[LLMModel] = None, llm_context_length: int = 8192):
        self.asr = asr_model
        self.embed = embed_model
        self.index_dir = index_dir
        self.llm = llm
        self.llm_context_length = max(1024, int(llm_context_length))
        self.llm_segments_budget = max(512, self.llm_context_length // 3)
        self.llm_merge_max_gap_sec = 1.0
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

        segments_for_windows = segments
        if self.llm and segments:
            report(65, "Refining segments with LLM...")
            metadata = self._build_llm_metadata(audio_path, podcast_name, audio_id)
            segments_for_windows = self._refine_segments_with_llm(
                segments,
                metadata,
                progress_callback=lambda p, m: report(65 + int(p * 0.05), m)
            )
            with open(os.path.join(audio_dir, "llm_segements.json"), "w", encoding="utf-8") as f:
                json.dump(segments_for_windows, f, ensure_ascii=False, indent=2)

        # 3. Create Windows (Story chunks)
        report(70, f"Creating windows (size={window_size}, stride={stride})...")
        windows = self._create_windows(segments_for_windows, window_size=window_size, stride=stride)

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

    def _build_llm_metadata(self, audio_path: str, podcast_name: str, audio_id: str) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "podcast_name": podcast_name,
            "audio_id": audio_id,
            "audio_file": os.path.basename(audio_path)
        }
        sidecar_path = f"{audio_path}.meta.json"
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    sidecar = json.load(f)
                if isinstance(sidecar, dict):
                    for key in ["title", "podcast", "description", "pubDate"]:
                        if key in sidecar:
                            metadata[key] = sidecar[key]
            except Exception:
                pass
        return metadata

    def _extract_json(self, text: str) -> Optional[Any]:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            block = match.group(1)
            try:
                return json.loads(block)
            except Exception:
                return None
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
        return None

    def _segment_payload_item(self, seg: Dict[str, Any], idx: int) -> Dict[str, Any]:
        return {
            "index": idx,
            "speaker": seg.get("speaker", ""),
            "text": seg.get("text", ""),
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0)
        }

    def _merge_adjacent_segments_for_llm(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not segments:
            return []
        merged: List[Dict[str, Any]] = []
        for seg in segments:
            if not merged:
                merged.append(dict(seg))
                continue
            prev = merged[-1]
            same_speaker = str(prev.get("speaker", "")).strip() == str(seg.get("speaker", "")).strip()
            prev_end = float(prev.get("end", 0.0) or 0.0)
            curr_start = float(seg.get("start", 0.0) or 0.0)
            close_enough = (curr_start - prev_end) <= self.llm_merge_max_gap_sec
            if same_speaker and close_enough:
                prev_text = str(prev.get("text", "")).strip()
                curr_text = str(seg.get("text", "")).strip()
                prev["text"] = f"{prev_text} {curr_text}".strip()
                prev["end"] = seg.get("end", prev.get("end"))
            else:
                merged.append(dict(seg))
        return merged

    def _split_segments_by_budget(self, segments: List[Dict[str, Any]]) -> List[tuple[int, int, List[Dict[str, Any]]]]:
        chunks: List[tuple[int, int, List[Dict[str, Any]]]] = []
        start = 0
        total = len(segments)
        while start < total:
            payload_segments: List[Dict[str, Any]] = []
            current_len = 2
            end = start
            while end < total:
                item = self._segment_payload_item(segments[end], end)
                item_len = len(json.dumps(item, ensure_ascii=False))
                projected = current_len + item_len + (1 if payload_segments else 0)
                if payload_segments and projected > self.llm_segments_budget:
                    break
                payload_segments.append(item)
                current_len = projected
                end += 1
                if projected >= self.llm_segments_budget:
                    break
            if not payload_segments:
                payload_segments.append(self._segment_payload_item(segments[start], start))
                end = start + 1
            chunks.append((start, end, payload_segments))
            start = end
        return chunks

    def _refine_segments_with_llm(self, segments: List[Dict[str, Any]], metadata: Dict[str, Any], progress_callback=None) -> List[Dict[str, Any]]:
        if not self.llm or not segments:
            return segments

        compact_segments = self._merge_adjacent_segments_for_llm(segments)
        total = len(compact_segments)
        refined = []
        chunks = self._split_segments_by_budget(compact_segments)

        for start, end, payload_segments in chunks:
            chunk = compact_segments[start:end]

            prompt = (
                "请根据播客元信息和分段文本执行以下任务："
                "1) speaker映射修正；2)去除无意义语气词与重复口头禅；3)结合元信息进行专有名词纠错。"
                "要求保留每条分段的index、start、end，不要合并或拆分分段。"
                "输出必须是JSON数组，每项包含index,speaker,text三个字段。"
                "\n播客元信息:\n"
                f"{json.dumps(metadata, ensure_ascii=False)}"
                "\n分段:\n"
                f"{json.dumps(payload_segments, ensure_ascii=False)}"
            )
            mapped: Dict[int, Dict[str, Any]] = {}
            try:
                response = self.llm.generate(prompt)
                parsed = self._extract_json(response)
                items = parsed.get("segments") if isinstance(parsed, dict) else parsed
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        idx = item.get("index")
                        if isinstance(idx, int) and start <= idx < end:
                            mapped[idx] = item
            except Exception as e:
                tqdm.write(f"LLM segment refine failed at chunk {start}-{end}: {e}")

            for idx, seg in enumerate(chunk):
                abs_idx = start + idx
                item = mapped.get(abs_idx, {})
                new_seg = dict(seg)
                new_text = item.get("text")
                new_speaker = item.get("speaker")
                if isinstance(new_text, str) and new_text.strip():
                    new_seg["raw_text"] = seg.get("text", "")
                    new_seg["text"] = new_text.strip()
                if new_speaker not in (None, ""):
                    new_seg["raw_speaker"] = seg.get("speaker", "")
                    new_seg["speaker"] = new_speaker
                refined.append(new_seg)

            if progress_callback:
                progress_callback(end / total, f"Refining segments {end}/{total}")

        return refined

    def test_llm_refine_segments(self, segments: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.llm:
            raise ValueError("LLM is not initialized")
        meta = dict(metadata or {})
        if "podcast_name" not in meta:
            meta["podcast_name"] = "test_podcast"
        if "audio_id" not in meta:
            meta["audio_id"] = "test_audio"
        return self._refine_segments_with_llm(segments, meta)


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
