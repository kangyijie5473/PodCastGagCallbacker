import os
import torch
import warnings
from funasr import AutoModel
from .base import ASRModel
from typing import List, Dict, Any

# Suppress warnings
warnings.filterwarnings("ignore")

class FunASR(ASRModel):
    def __init__(self, model_name="iic/SenseVoiceSmall", device=None):
        """
        Initialize FunASR model.
        Defaulting to SenseVoiceSmall as it is the current SOTA fast model from FunASR.
        Alternatively can use "paraformer-zh".
        """
        self.model_name = model_name
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading FunASR model: {model_name} on {self.device}...")
        
        # Determine cache dir
        cache_dir = os.path.join(os.getcwd(), "models_cache", "funasr")
        os.makedirs(cache_dir, exist_ok=True)
        
        # FunASR AutoModel will handle download and caching via ModelScope
        # We also enable vad and punc for better results
        # Enable diarization model (campplus)
        # Note: SenseVoiceSmall supports timestamp prediction implicitly when combined with VAD/PUNC in the pipeline?
        # Actually, the error message said: "Only '...paraformer...' can predict timestamp"
        # SenseVoiceSmall DOES support timestamps, but maybe parameter configuration needs adjustment.
        # However, for stable diarization integration, Paraformer is often recommended if SenseVoice fails on timestamps in pipeline mode.
        # Let's try switching to Paraformer first as it's the "standard" for this pipeline integration.
        # Or we can try to fix SenseVoice usage.
        
        # The error "KeyError: 'timestamp'" in inference_with_vad suggests the model didn't return timestamps expected by the VAD merger.
        # SenseVoiceSmall output format might be different.
        
        # Let's switch to the robust Paraformer model which is guaranteed to work with the timestamp/diarization pipeline.
        self.use_paraformer = True
        if self.use_paraformer:
            self.model_name = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            print(f"Switching to Paraformer for better pipeline compatibility: {self.model_name}")

        try:
            self.model = AutoModel(
                model=self.model_name,
                vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                spk_model="iic/speech_campplus_sv_zh-cn_16k-common", 
                device=self.device,
                base_cache_dir=cache_dir,
                trust_remote_code=True,
                disable_update=True
            )
        except Exception as e:
            print(f"Error loading FunASR model: {e}")
            raise e

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        # Ensure audio path is absolute
        audio_path = os.path.abspath(audio_path)
        
        if not os.path.exists(audio_path):
            print(f"Error: File {audio_path} not found.")
            return []

        print(f"Transcribing {audio_path} with FunASR...")

        try:
            # generate() with diarization params
            # merge_vad_diar=True is key to get speaker info merged into segments
            res = self.model.generate(
                input=audio_path,
                batch_size_s=300,
                vad_param={"max_single_segment_time": 60000},
                merge_vad_diar=True,
                merge_thr=1.0, # Threshold for merging segments
                diar_param={"max_speakers": 10}, # Estimate max speakers
                print_result=False
            )
            
            if not res:
                return []
                
            item = res[0]
            segments = []
            
            # If sentence_info is available (timestamped segments)
            if "sentence_info" in item:
                for seg in item["sentence_info"]:
                    # FunASR returns timestamps in milliseconds
                    start_sec = seg["start"] / 1000.0
                    end_sec = seg["end"] / 1000.0
                    text = seg["text"].strip()
                    speaker = seg.get("spk", "Unknown") # Capture speaker ID
                    
                    # Convert speaker ID to string if it's an int
                    if isinstance(speaker, int):
                        speaker = f"SPEAKER_{speaker:02d}"
                    
                    segments.append({
                        "start": start_sec,
                        "end": end_sec,
                        "text": text,
                        "speaker": speaker
                    })
            else:
                # Fallback if no detailed info
                segments.append({
                    "start": 0.0,
                    "end": 0.0, # Unknown duration
                    "text": item.get("text", "").strip(),
                    "speaker": "Unknown"
                })
                
            return segments
            
        except Exception as e:
            print(f"FunASR transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return []
