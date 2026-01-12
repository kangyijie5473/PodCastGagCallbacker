from .base import ASRModel
from typing import List, Dict, Any
import whisper
import warnings
import ssl

# Hack to bypass SSL verification for Whisper download if needed
ssl._create_default_https_context = ssl._create_unverified_context

class LocalWhisperASR(ASRModel):
    def __init__(self, model_size="tiny"):
        # Suppress warnings for cleaner output
        # warnings.filterwarnings("ignore")
        self.model = whisper.load_model(model_size)
    
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        # Whisper transcribe returns dict with 'segments'
        # Ensure fp16=False for compatibility with CPU/various hardware if needed
        result = self.model.transcribe(audio_path, fp16=False)
        return [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            }
            for seg in result["segments"]
        ]
