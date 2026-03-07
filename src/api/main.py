from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from src.models.embedding import LocalEmbedding
from src.services.searcher import SearchService

app = FastAPI(title="Podcast Search API")

DATA_DIR = "data"
embed_model = None
search_service = None

@app.on_event("startup")
def load_models():
    global embed_model, search_service
    print("Loading models for API...")
    # Lazy loading or eager loading? Eager for API.
    embed_model = LocalEmbedding()
    search_service = SearchService(embed_model, DATA_DIR)
    print("Models loaded.")

class SearchResult(BaseModel):
    score: float
    audio_id: str
    start: float
    end: float
    text: str

@app.get("/search", response_model=List[SearchResult])
def search(q: str, id: Optional[str] = None, top_k: int = 5):
    if not search_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    results = search_service.search(q, audio_id=id, top_k=top_k)
    return [
        SearchResult(
            score=r["score"],
            audio_id=r["audio_id"],
            start=r["window"]["start"],
            end=r["window"]["end"],
            text=r["window"]["text"]
        )
        for r in results
    ]

@app.get("/indices")
def list_indices():
    if not search_service:
        return []
    return search_service.list_indices()

@app.get("/health")
def health():
    return {"status": "ok"}
