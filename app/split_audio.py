import os
import numpy as np
import librosa
import soundfile as sf

def split_audio(audio_path, out_dir, chunk_seconds=600, overlap_seconds=0, sr=16000):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    os.makedirs(out_dir, exist_ok=True)
    chunk = int(chunk_seconds * sr)
    overlap = int(overlap_seconds * sr)
    res = []
    i = 0
    s = 0
    n = len(y)
    while s < n:
        e = min(s + chunk, n)
        clip = y[s:e]
        fn = f"chunk_{i:03d}.wav"
        fp = os.path.join(out_dir, fn)
        sf.write(fp, clip, sr)
        res.append({"file": fp, "start": s / sr, "end": e / sr})
        i += 1
        if e == n:
            break
        if overlap > 0:
            s = s + chunk - overlap
        else:
            s = s + chunk
    return res
if __name__ == "__main__":
    audio_path = "/Users/kangkang/code/PodCastGagCallbacker/debug/zhengjingbaba.mp4"
    out_dir = "/Users/kangkang/code/PodCastGagCallbacker/debug//splits"
    split_audio(audio_path, out_dir)