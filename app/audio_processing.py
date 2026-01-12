import os
import json
import numpy as np
import librosa

def transcribe_audio(audio_path, model_size="tiny"):
    import whisper
    y, _ = librosa.load(audio_path, sr=16000, mono=True)
    y = np.asarray(y, dtype=np.float32)
    model = whisper.load_model(model_size)
    result = model.transcribe(y, fp16=False)
    segments = []
    for seg in result.get("segments", []):
        segments.append({"text": seg.get("text", "").strip(), "start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0))})
    return segments

def _load_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    return y, sr

def assign_speakers(audio_path, segments, similarity_threshold=0.75):
    from resemblyzer import VoiceEncoder
    y, sr = _load_audio(audio_path)
    encoder = VoiceEncoder()
    centroids = []
    labels = []
    for seg in segments:
        s = max(int(seg["start"] * sr), 0)
        e = min(int(seg["end"] * sr), len(y))
        clip = y[s:e]
        if len(clip) == 0:
            labels.append(0)
            continue
        emb = encoder.embed_utterance(clip)
        if not centroids:
            centroids.append(emb)
            labels.append(0)
            continue
        sims = [float(np.dot(emb, c) / (np.linalg.norm(emb) * np.linalg.norm(c))) for c in centroids]
        i = int(np.argmax(sims))
        if sims[i] >= similarity_threshold:
            centroids[i] = (centroids[i] + emb) / 2.0
            labels.append(i)
        else:
            centroids.append(emb)
            labels.append(len(centroids) - 1)
    return labels

def detect_laughter(audio_path, segments):
    y, sr = _load_audio(audio_path)
    scores = []
    for seg in segments:
        t = seg["text"]
        s = max(int(seg["start"] * sr), 0)
        e = min(int(seg["end"] * sr), len(y))
        clip = y[s:e]
        if len(clip) == 0:
            scores.append(0.0)
            continue
        zcr = librosa.feature.zero_crossing_rate(y=clip)
        rms = librosa.feature.rms(y=clip)
        a = float(zcr.mean())
        b = float(rms.mean())
        c = 0.0
        if ("哈哈" in t) or ("笑" in t) or ("laughter" in t) or ("laugh" in t):
            c = 1.0
        elif a > 0.1 and b > 0.03:
            c = 0.5
        else:
            c = 0.0
        scores.append(c)
    return scores

def build_index(audio_path, segments, speakers, laughs, index_dir):
    os.makedirs(index_dir, exist_ok=True)
    data = []
    texts = []
    for i, seg in enumerate(segments):
        item = {
            "i": i,
            "text": seg["text"],
            "start": seg["start"],
            "end": seg["end"],
            "speaker": int(speakers[i]),
            "laughter": float(laughs[i]),
            "audio_path": audio_path
        }
        data.append(item)
        texts.append(seg["text"]) 
    embs = None
    method = "st"
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        embs = model.encode(texts, normalize_embeddings=True)
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import pickle
        vec = TfidfVectorizer(max_features=2048)
        X = vec.fit_transform(texts)
        embs = X.toarray().astype(np.float32)
        with open(os.path.join(index_dir, "tfidf.pkl"), "wb") as f:
            pickle.dump(vec, f)
        method = "tfidf"
    np.save(os.path.join(index_dir, "embeddings.npy"), np.array(embs))
    with open(os.path.join(index_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"items": data, "method": method}, f, ensure_ascii=False)
    return os.path.join(index_dir, "index.json"), os.path.join(index_dir, "embeddings.npy")
