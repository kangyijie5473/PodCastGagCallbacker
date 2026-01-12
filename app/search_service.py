import os
import json
import numpy as np

class SearchService:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.data = None
        self.embs = None
        self.model = None
        self.audio_path = None
        self.method = None
        self._load()

    def _load(self):
        idx = os.path.join(self.index_dir, "index.json")
        emb = os.path.join(self.index_dir, "embeddings.npy")
        if os.path.exists(idx) and os.path.exists(emb):
            with open(idx, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if isinstance(meta, dict) and "items" in meta:
                    self.data = meta["items"]
                    self.method = meta.get("method")
                else:
                    self.data = meta
            self.embs = np.load(emb)
            if self.data:
                self.audio_path = self.data[0].get("audio_path")

    def _ensure_model(self):
        if self.model is None:
            tfidf_pkl = os.path.join(self.index_dir, "tfidf.pkl")
            if os.path.exists(tfidf_pkl):
                import pickle
                with open(tfidf_pkl, "rb") as f:
                    self.model = pickle.load(f)
                self.method = "tfidf"
            else:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
                self.method = "st"

    def search(self, q, top_k=5):
        if self.data is None or self.embs is None:
            return []
        self._ensure_model()
        if self.method == "tfidf":
            X = self.model.transform([q])
            qemb = X.toarray().astype(np.float32)[0]
        else:
            qemb = self.model.encode([q], normalize_embeddings=True)[0]
        sims = self.embs @ qemb
        sims = sims.tolist()
        scores = []
        for i, s in enumerate(sims):
            r = self.data[i]
            sc = float(s) + 0.2 * float(r.get("laughter", 0.0))
            scores.append((sc, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        res = []
        for sc, i in scores[:top_k]:
            r = self.data[i]
            res.append({
                "text": r["text"],
                "start": r["start"],
                "end": r["end"],
                "speaker": r["speaker"],
                "laughter": r["laughter"],
                "score": sc,
            })
        return res
