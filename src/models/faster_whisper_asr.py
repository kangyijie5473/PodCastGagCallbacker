import os
import time
import torch
import warnings
import soundfile as sf
import subprocess
import numpy as np
import traceback
from typing import List, Dict, Any
from faster_whisper import WhisperModel
from .mock_audio_decoder import MockAudioDecoder

try:
    import pyannote.audio.core.io
    # print("Injecting MockAudioDecoder (soundfile-based) into pyannote.audio.core.io")
    pyannote.audio.core.io.AudioDecoder = MockAudioDecoder
except ImportError:
    pass
# ---------------------------------------------

from pyannote.audio import Pipeline
from pyannote.core import Segment
from .base import ASRModel

# Suppress some warnings
warnings.filterwarnings("ignore")

class FasterWhisperASR(ASRModel):
    def __init__(self, model_size="large-v3", device=None, compute_type="float16"):
        """
        Initialize Faster Whisper ASR model and Pyannote Diarization pipeline.
        
        Args:
            model_size (str): Whisper model size (default: "large-v3")
            device (str): Device to run on ("cuda" or "cpu"). If None, auto-detect.
            compute_type (str): Compute type for Whisper ("float16", "int8_float16", "int8", etc.)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.compute_type = compute_type if self.device == "cuda" else "int8"
        
        # Determine download root
        download_root = os.path.join(os.getcwd(), "models_cache", "faster-whisper")
        os.makedirs(download_root, exist_ok=True)
        
        print(f"Loading Faster Whisper model: {model_size} on {self.device} with {self.compute_type}...")
        try:
            self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type, download_root=download_root)
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise e

        print("Loading Pyannote Diarization pipeline...")
        # Check for HF_TOKEN
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("WARNING: HF_TOKEN environment variable not set. Pyannote diarization might fail if using gated models.")
            
        try:
            if hf_token:
                from huggingface_hub import login
                try:
                    login(token=hf_token)
                except Exception as e:
                    print(f"Failed to login to Hugging Face: {e}")

            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
            if self.diarization_pipeline:
                self.diarization_pipeline.to(torch.device(self.device))
        except Exception as e:
            print(f"Error loading Pyannote pipeline: {e}")
            if "403" in str(e) or "401" in str(e):
                print("\n" + "!"*50)
                print("CRITICAL: Authentication failed for Pyannote model.")
                print("If you are using a Fine-grained Access Token, please ensure you have enabled:")
                print("   'Read access to contents of all public gated repositories'")
                print("in your token permissions settings on Hugging Face.")
                print("!"*50 + "\n")
            else:
                print("Please ensure you have accepted the user conditions on Hugging Face for pyannote/speaker-diarization-3.1")
                print("and set the HF_TOKEN environment variable.")
            self.diarization_pipeline = None

    def _convert_to_wav(self, input_path: str, output_path: str):
        """Convert audio to WAV using ffmpeg (simple conversion)."""
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            output_path
        ]
        # Run ffmpeg quietly
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe audio file and perform speaker diarization.
        This simplified version removes complex stereo handling and directly processes the file.
        It ensures a compatible WAV format for Pyannote but uses the original file for Whisper ASR.
        """
        # Ensure audio path is absolute
        audio_path = os.path.abspath(audio_path)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        print(f"Transcribing {audio_path}...")
        pipeline_start = time.perf_counter()
        
        # Prepare Debug Dir for intermediate files
        debug_dir = os.path.join(os.getcwd(), "debug_audio")
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.basename(audio_path)
        
        # Ensure 16k mono WAV exists for Pyannote/SoundFile compatibility
        # (SoundFile often fails on MP4/AAC, so we convert to WAV)
        wav_path = os.path.join(debug_dir, f"{base_name}.wav")
        converted = False
        convert_ms = 0.0
        
        try:
            print(f"Ensuring WAV format -> {wav_path}...")
            t_convert = time.perf_counter()
            self._convert_to_wav(audio_path, wav_path)
            convert_ms = (time.perf_counter() - t_convert) * 1000
            converted = True
        except Exception as e:
            print(f"Conversion failed, trying original file: {e}")
            wav_path = audio_path # Fallback
        
        # 1. Transcribe with Faster Whisper
        # USE ORIGINAL AUDIO for Whisper to avoid quality loss from conversion if possible
        # Whisper handles MP4/AAC natively via ffmpeg usually.
        print(f"Using ORIGINAL audio for Whisper: {audio_path}")
        asr_start = time.perf_counter()
        try:
            segments_generator, info = self.model.transcribe(audio_path, beam_size=5, language='zh', vad_filter=True)
            whisper_segments = list(segments_generator)
            asr_ms = (time.perf_counter() - asr_start) * 1000
            print(f"Transcription complete. Found {len(whisper_segments)} segments.")
            print(f"[Timing] ASR(Whisper)={asr_ms:.1f}ms convert={convert_ms:.1f}ms")
            
            # Print first few segments for debugging
            for i in range(min(5, len(whisper_segments))):
                print(f"  [{whisper_segments[i].start:.2f}-{whisper_segments[i].end:.2f}]: {whisper_segments[i].text}")
        except Exception as e:
            asr_ms = (time.perf_counter() - asr_start) * 1000
            print(f"Whisper transcription failed: {e}")
            print(f"[Timing] ASR(Whisper) failed after {asr_ms:.1f}ms convert={convert_ms:.1f}ms")
            print(
                "Whisper context:",
                {
                    "audio_path": audio_path,
                    "device": self.device,
                    "compute_type": self.compute_type,
                    "wav_path": wav_path,
                    "converted": converted
                }
            )
            if torch.cuda.is_available():
                try:
                    print(
                        "CUDA memory:",
                        {
                            "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
                            "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
                            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
                            "max_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 / 1024, 2)
                        }
                    )
                except Exception as mem_e:
                    print(f"CUDA memory inspection failed: {mem_e}")
            traceback.print_exc()
            return []

        # 2. Diarization with Pyannote
        # USE CONVERTED WAV for Pyannote (required by soundfile/MockAudioDecoder)
        diarization_path = wav_path if converted else audio_path
        
        final_segments = []
        if self.diarization_pipeline:
            print(f"Running speaker diarization on {diarization_path}...")
            diar_start = time.perf_counter()
            try:
                # Run pipeline
                diarization = self.diarization_pipeline(diarization_path)
                
                # Handle Pyannote 3.1+ DiarizeOutput vs Annotation
                if hasattr(diarization, "speaker_diarization"):
                    # Some pipelines return a wrapper object
                    diarization = diarization.speaker_diarization
                 
                final_segments = self._merge_diarization(whisper_segments, diarization)
                diar_ms = (time.perf_counter() - diar_start) * 1000
                total_ms = (time.perf_counter() - pipeline_start) * 1000
                print(f"[Timing] Diarization(Pyannote)={diar_ms:.1f}ms total_pipeline={total_ms:.1f}ms")
            except Exception as e:
                diar_ms = (time.perf_counter() - diar_start) * 1000
                print(f"Diarization failed: {e}")
                print(f"[Timing] Diarization(Pyannote) failed after {diar_ms:.1f}ms")
                traceback.print_exc()
                # Fallback to just Whisper segments with "Unknown" speaker
                final_segments = self._fallback_segments(whisper_segments)
        else:
            print("Diarization pipeline not available, skipping.")
            final_segments = self._fallback_segments(whisper_segments)
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        print(f"[Timing] Transcribe pipeline total={total_ms:.1f}ms")
            
        return final_segments

    def _fallback_segments(self, whisper_segments):
        return [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "speaker": "Unknown"
            }
            for seg in whisper_segments
        ]

    def _merge_diarization(self, whisper_segments, diarization):
        """
        Merge Whisper segments with Pyannote diarization results.
        Assigns the most frequent speaker in the segment's time range.
        """
        result_segments = []
        
        for seg in whisper_segments:
            start = seg.start
            end = seg.end
            text = seg.text.strip()
            
            # Find speakers in this range
            speakers = {}
            
            # crop returns the part of the annotation included in the support
            from pyannote.core import Segment
            segment_region = Segment(start, end)
            
            # Get intersection
            overlap = diarization.crop(segment_region)
            
            # Calculate duration for each speaker in the overlap
            for turn, _, speaker in overlap.itertracks(yield_label=True):
                duration = turn.end - turn.start
                if speaker in speakers:
                    speakers[speaker] += duration
                else:
                    speakers[speaker] = duration
            
            # Find dominant speaker
            if speakers:
                dominant_speaker = max(speakers, key=speakers.get)
            else:
                dominant_speaker = "Unknown"
            
            result_segments.append({
                "start": start,
                "end": end,
                "text": text,
                "speaker": dominant_speaker
            })
            
        return result_segments
