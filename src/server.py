import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from src.models.faster_whisper_asr import FasterWhisperASR
from src.models.embedding import LocalEmbedding
try:
    from src.models.reranker import RerankerModel
except ImportError:
    RerankerModel = None
from src.models.llm import OpenAILLM
from src.services.indexer import IndexingService
from src.services.searcher import SearchService
from src.services.rag import RAGService
from src.services.collector import CollectorService
from src.services.downloader import PodcastDownloader

DATA_DIR = "data"
DOWNLOAD_DIR = "downloads"

# Global services
services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
    os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
    # Use Hugging Face Mirror if needed
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # Startup
    print("Loading models...")
    # Load environment variables
    load_dotenv()
    
    # Initialize models
    # We use a default config for now
    model_size = "large-v3"
    
    asr = FasterWhisperASR(model_size=model_size)
    embed = LocalEmbedding()
    
    # Initialize Reranker
    reranker = None
    if RerankerModel:
        try:
            print("Initializing Reranker...")
            reranker = RerankerModel()
        except Exception as e:
            print(f"Failed to initialize reranker: {e}")
    
    llm_base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    llm = None
    try:
        llm = OpenAILLM(api_key=None, base_url=llm_base_url, model=None)
        print(f"LLM initialized at {llm_base_url}")
    except Exception as e:
        print(f"Failed to initialize local LLM: {e}")
        llm = None
    
    indexer = IndexingService(asr, embed, DATA_DIR, llm=llm)
    collector = CollectorService(indexer)
    searcher = SearchService(embed, DATA_DIR, reranker=reranker)
    
    rag = None
    if llm:
        rag = RAGService(searcher, llm)
        
    services["indexer"] = indexer
    services["collector"] = collector
    services["searcher"] = searcher
    services["rag"] = rag
    services["downloader"] = PodcastDownloader(DOWNLOAD_DIR)
    
    yield
    
    # Shutdown
    print("Shutting down...")
    services.clear()

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# Models
class SearchRequest(BaseModel):
    query: str
    podcast_name: Optional[str] = None
    audio_id: Optional[str] = None
    top_k: int = 5
    use_rag: bool = False

class PodcastLinkRequest(BaseModel):
    url: str
    limit: int = 0

class IngestDirectoryRequest(BaseModel):
    source: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[dict]
    answer: Optional[str] = None

# Routes

@app.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    searcher = services["searcher"]
    rag = services.get("rag")
    
    if req.use_rag:
        if not rag:
            # RAG requested but not available
            results = searcher.search(req.query, podcast_name=req.podcast_name, audio_id=req.audio_id, top_k=req.top_k)
            return {
                "results": results, 
                "answer": "RAG service is not available. Please configure LLM_API_KEY in .env file."
            }
            
        results = searcher.search(req.query, podcast_name=req.podcast_name, audio_id=req.audio_id, top_k=req.top_k)
        answer = rag.answer(req.query, podcast_name=req.podcast_name, audio_id=req.audio_id, top_k=req.top_k, results=results)
        return {"results": results, "answer": answer}
    else:
        results = searcher.search(req.query, podcast_name=req.podcast_name, audio_id=req.audio_id, top_k=req.top_k)
        return {"results": results, "answer": None}

@app.post("/api/podcast/submit")
async def submit_podcast(
    req: PodcastLinkRequest,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(process_podcast_download, req.url, req.limit)
    return {"status": "success", "message": "Podcast download started"}

@app.post("/api/ingest")
async def ingest_directory(req: IngestDirectoryRequest):
    source = os.path.abspath(req.source or DOWNLOAD_DIR)
    if not os.path.isdir(source):
        raise HTTPException(status_code=400, detail=f"Source directory not found: {source}")

    collector = services["collector"]
    processed = []
    default_downloads = os.path.abspath(DOWNLOAD_DIR)

    if source == default_downloads:
        for podcast_name in sorted(os.listdir(source)):
            full_path = os.path.join(source, podcast_name)
            if os.path.isdir(full_path):
                collector.sync_podcast(full_path, podcast_name)
                processed.append({"podcast_name": podcast_name, "source": full_path})
    else:
        podcast_name = os.path.basename(os.path.normpath(source)) or "default"
        collector.sync_podcast(source, podcast_name)
        processed.append({"podcast_name": podcast_name, "source": source})

    return {
        "status": "success",
        "message": "Directory processed",
        "processed": processed
    }

@app.get("/api/podcasts")
async def list_podcasts():
    searcher = services["searcher"]
    # List indices returns flattened list [{"podcast":..., "audio_id":...}]
    indices = searcher.list_indices()
    
    # Organize hierarchy
    podcasts = {}
    for item in indices:
        p_name = item["podcast"]
        if p_name not in podcasts:
            podcasts[p_name] = {"name": p_name, "episodes": []}
        
        # Add basic episode info
        podcasts[p_name]["episodes"].append({
            "id": item["audio_id"],
            "indexed_at": "Unknown" # Metadata could be enhanced later
        })
    
    return list(podcasts.values())

@app.get("/api/audio/{podcast_name}/{audio_id}")
async def get_audio(podcast_name: str, audio_id: str):
    # Only serve from Downloads directory
    # Structure: downloads/{podcast_name}/{filename}
    # The audio_id is typically the filename without extension.
    
    podcast_dir = os.path.join(DOWNLOAD_DIR, podcast_name)
    if os.path.exists(podcast_dir):
        for f in os.listdir(podcast_dir):
            # Check if file starts with audio_id and is an audio/video file (not json/txt)
            if f.startswith(audio_id) and not f.endswith(".json") and not f.endswith(".txt"):
                file_path = os.path.join(podcast_dir, f)
                # Determine media type roughly
                media_type = "audio/mp4" # Default fallback
                if f.endswith(".mp3"): media_type = "audio/mpeg"
                elif f.endswith(".wav"): media_type = "audio/wav"
                elif f.endswith(".m4a"): media_type = "audio/mp4"
                
                return FileResponse(file_path, media_type=media_type)

    raise HTTPException(status_code=404, detail=f"Audio file not found in downloads/{podcast_name}")

def process_podcast_download(url: str, limit: int):
    print(f"Downloading podcast from {url}")
    downloader = services["downloader"]
    try:
        # Determine podcast name from URL or metadata?
        # Downloader saves to downloads/{podcast_title} usually.
        # We need to capture the destination directory to run ingest.
        # For now, PodcastDownloader.download returns nothing.
        # We might need to modify Downloader to return the saved path or just scan downloads dir.
        
        # Let's assume standard flow:
        # 1. Download
        # 2. Find where it went (or just scan all downloads?)
        # Ideally Downloader should return the path.
        
        # NOTE: Modifying Downloader to return path would be good. 
        # For now, we'll just download. The user might need to trigger ingest manually or we auto-scan downloads.
        # Let's try to infer or improve Downloader later. 
        # For this turn, let's just download.
        downloader.download(url, limit=limit)
        
        # Auto-ingest? We don't know the folder name easily without parsing again.
        # We will leave auto-ingest for next step or assume user does it.
        # OR: we can list directories in downloads/ and see which one is new.
        # Let's keep it simple: just download.
        print(f"Download complete for {url}")
        
        # Trigger ingest for all in downloads? Might be expensive.
        # Let's try to ingest 'downloads' recursively?
        # The CLI ingest takes a name and source.
        # We can iterate over subdirs in downloads/ and ingest them.
        for podcast_dir in os.listdir(DOWNLOAD_DIR):
            full_path = os.path.join(DOWNLOAD_DIR, podcast_dir)
            if os.path.isdir(full_path):
                collector = services["collector"]
                collector.sync_podcast(full_path, podcast_dir)
                
    except Exception as e:
        print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5473)
