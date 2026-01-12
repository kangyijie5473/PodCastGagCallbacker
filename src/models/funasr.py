import os
from funasr import AutoModel
from .base import ASRModel
from typing import List, Dict, Any

class LocalFunASR(ASRModel):
    def __init__(self, model_name="paraformer-zh"):
        self.model_name = model_name
        
        # FunASR AutoModel will handle download and caching via ModelScope
        # We also enable vad and punc for better results
        # Enable diarization model (campplus)
        self.model = AutoModel(
            model=self.model_name,
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="iic/speech_campplus_sv_zh-cn_16k-common", 
        )

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        # Ensure audio path is absolute
        audio_path = os.path.abspath(audio_path)
        
        # generate() with diarization params
        # Note: output_dir is optional, but merge_vad_diar=True is key
        res = self.model.generate(
            input=audio_path,
            batch_size_s=300,
            vad_param={"max_single_segment_time": 30000},
            merge_vad_diar=True,  # Merge VAD and Diarization
            diar_param={"max_speakers": 15}, # 大概4 5个主播，和10个观众
            print_result=False
        )
        
        if not res:
            return []
            
        item = res[0]
        segments = []
        
        # If sentence_info is available (timestamped segments)
        if "sentence_info" in item:
            for seg in item["sentence_info"]:
                segments.append({
                    "start": seg["start"] / 1000.0,
                    "end": seg["end"] / 1000.0,
                    "text": seg["text"].strip(),
                    "speaker": seg.get("spk", 0) # Capture speaker ID
                })
        else:
            # Fallback
            segments.append({
                "start": 0.0,
                "end": 0.0,
                "text": item.get("text", "").strip(),
                "speaker": 0
            })
            
        return segments

