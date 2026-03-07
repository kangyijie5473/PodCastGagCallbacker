import os
import shutil
import asyncio
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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
UPLOAD_DIR = "uploads"
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
    
    # Optional: Load LLM if env vars are present, or lazy load
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL")
    llm_model = os.getenv("LLM_MODEL", "doubao-seed-1-8-251228")
    
    llm = None
    if llm_api_key:
        print(f"Initializing LLM ({llm_model})...")
        llm = OpenAILLM(api_key=llm_api_key, base_url=llm_base_url, model=llm_model)
    else:
        print("WARNING: LLM_API_KEY not found. RAG and Refine features will be disabled.")
    
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

import uuid

# Global tasks store
tasks = {}

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.post("/api/upload")
async def upload_audio(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = None
):
    # Save file
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "id": task_id,
        "status": "pending",
        "progress": 0,
        "message": "Queued",
        "filename": file.filename
    }
        
    # Trigger ingestion in background
    # For user uploads, we can use "user_uploads" as podcast_name
    background_tasks.add_task(process_upload, file_path, "user_uploads", task_id)
    
    return {
        "status": "success", 
        "message": "File uploaded and processing started", 
        "filename": file.filename,
        "task_id": task_id
    }

@app.post("/api/podcast/submit")
async def submit_podcast(
    req: PodcastLinkRequest,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(process_podcast_download, req.url, req.limit)
    return {"status": "success", "message": "Podcast download started"}

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

# Background Tasks

def process_upload(file_path: str, podcast_name: str, task_id: str):
    print(f"Processing upload: {file_path}")
    indexer = services["indexer"]
    audio_id = os.path.splitext(os.path.basename(file_path))[0]
    
    tasks[task_id]["status"] = "processing"
    tasks[task_id]["message"] = "Starting..."
    
    def update_progress(p, msg):
        tasks[task_id]["progress"] = p
        tasks[task_id]["message"] = msg
    
    try:
        indexer.process_audio(file_path, audio_id, podcast_name=podcast_name, progress_callback=update_progress)
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = "Done"
        print(f"Finished processing {file_path}")
    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["message"] = str(e)
        print(f"Error processing {file_path}: {e}")

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
    uvicorn.run(app, host="127.0.0.1", port=5473)
