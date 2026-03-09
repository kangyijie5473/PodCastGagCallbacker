import argparse
import os
import sys
import requests
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Lazy imports to avoid loading heavy models unnecessarily
def get_asr_model(model_size):
    from src.models.faster_whisper_asr import FasterWhisperASR
    return FasterWhisperASR(model_size=model_size)

def get_embedding_model():
    from src.models.embedding import LocalEmbedding
    return LocalEmbedding()

def get_llm_model(api_key, base_url, model):
    from src.models.llm import OpenAILLM
    return OpenAILLM(api_key=api_key, base_url=base_url, model=model)

from src.services.indexer import IndexingService
from src.services.collector import CollectorService
from src.services.downloader import PodcastDownloader

DATA_DIR = "data"
DEFAULT_SERVER_URL = "http://127.0.0.1:5473"

def main():
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
    os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
    # Use Hugging Face Mirror if needed
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    parser = argparse.ArgumentParser(description="Podcast Search Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Process Command (Local)
    proc_parser = subparsers.add_parser("process", help="Process audio file locally")
    proc_parser.add_argument("--audio_path", help="Path to audio file")
    proc_parser.add_argument("--id", help="Unique ID for the audio (default: filename)")
    proc_parser.add_argument("--model-size", default="large-v3", help="Faster Whisper model size")
    proc_parser.add_argument("--window-size", type=int, default=20, help="Window size in segments")
    proc_parser.add_argument("--stride", type=int, default=5, help="Window stride in segments")
    proc_parser.add_argument("--refine", action="store_true", help="Refine transcript with LLM")
    proc_parser.add_argument("--api-key", help="LLM API Key")
    proc_parser.add_argument("--base-url", help="LLM Base URL")
    proc_parser.add_argument("--model", default="doubao-seed-1-8-251228", help="LLM Model name")
    
    # Search Command (API)
    search_parser = subparsers.add_parser("search", help="Search via API")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--name", help="Podcast Name to filter by")
    search_parser.add_argument("--id", help="Audio ID to search in (optional)")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("--server", default=DEFAULT_SERVER_URL, help="API Server URL")
    
    # RAG Command (API)
    rag_parser = subparsers.add_parser("rag", help="Ask a question via API (RAG)")
    rag_parser.add_argument("--query", required=True, help="Question")
    rag_parser.add_argument("--name", help="Podcast Name to filter by")
    rag_parser.add_argument("--id", help="Audio ID (optional)")
    rag_parser.add_argument("--top-k", type=int, default=5)
    rag_parser.add_argument("--server", default=DEFAULT_SERVER_URL, help="API Server URL")

    # Ingest Command (Local)
    ingest_parser = subparsers.add_parser("ingest", help="Ingest/Sync podcast directory locally")
    ingest_parser.add_argument("--name", required=True, help="Podcast Name (e.g. 'tech_talk')")
    ingest_parser.add_argument("--source", required=True, help="Source directory containing audio files")
    ingest_parser.add_argument("--window-size", type=int, default=20)
    ingest_parser.add_argument("--stride", type=int, default=5)
    ingest_parser.add_argument("--refine", action="store_true", help="Refine transcript with LLM")
    ingest_parser.add_argument("--api-key", help="LLM API Key")
    ingest_parser.add_argument("--base-url", help="LLM Base URL")
    ingest_parser.add_argument("--model", default="doubao-seed-1-8-251228", help="LLM Model name")
    
    # Download Command (Local)
    download_parser = subparsers.add_parser("download", help="Download podcast from URL (RSS or XiaoYuZhou)")
    download_parser.add_argument("url", help="RSS Feed URL or XiaoYuZhou Podcast Page URL")
    download_parser.add_argument("--limit", type=int, default=0, help="Limit number of episodes to download (0 for all)")
    download_parser.add_argument("--dest", default="downloads", help="Destination directory")
    download_parser.add_argument("--metadata-only", action="store_true", help="Only download metadata (JSON), skip audio")

    args = parser.parse_args()

    if args.command == "process":
        # ... (keep existing process logic)
        if not os.path.exists(args.audio_path):
            print(f"Error: File {args.audio_path} not found.")
            return

        audio_id = args.id or os.path.splitext(os.path.basename(args.audio_path))[0]
        
        print(f"Loading models (Faster Whisper: {args.model_size}, Embedding: bge-base-zh-v1.5)...")
        # Initialize models
        llm = None
        if args.refine:
            llm = get_llm_model(args.api_key, args.base_url, args.model)
            
        asr = get_asr_model(args.model_size)
        embed = get_embedding_model()
        
        indexer = IndexingService(asr, embed, DATA_DIR, llm=llm)
        res = indexer.process_audio(args.audio_path, audio_id, window_size=args.window_size, stride=args.stride)
        
        print("Processing result:", res)
        print(f"Debug data saved to {os.path.join(DATA_DIR, audio_id)}")
        
    elif args.command == "search":
        url = f"{args.server}/api/search"
        payload = {
            "query": args.query,
            "podcast_name": args.name,
            "audio_id": args.id,
            "top_k": args.top_k,
            "use_rag": False
        }
        
        try:
            print(f"Sending request to {url}...")
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            results = data.get("results", [])
            print(f"\nTop {len(results)} results for '{args.query}':")
            print("="*60)
            for i, res in enumerate(results):
                score = res.get("score", 0.0)
                metadata = res.get("metadata", {})
                text = res.get("text", "")
                window_id = res.get("window_id", "N/A")
                podcast = metadata.get("podcast", "Unknown")
                audio_id = metadata.get("audio_id", "Unknown")
                start = metadata.get("start", 0.0)
                end = metadata.get("end", 0.0)
                
                print(f"{i+1}. [Score: {score:.4f}] {podcast} - {audio_id} (ID: {window_id})")
                print(f"   Time: {start:.1f}s - {end:.1f}s")
                print(f"   Text: {text[:200]}..." if len(text) > 200 else f"   Text: {text}")
                print("-" * 60)
                
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to server at {args.server}. Is it running?")
        except Exception as e:
            print(f"Error during search: {e}")

    elif args.command == "rag":
        url = f"{args.server}/api/search"
        payload = {
            "query": args.query,
            "podcast_name": args.name,
            "audio_id": args.id,
            "top_k": args.top_k,
            "use_rag": True
        }
        
        try:
            print(f"Sending request to {url}...")
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            
            answer = data.get("answer", "No answer returned.")
            print("\n" + "="*20 + " ANSWER " + "="*20)
            print(answer)
            print("="*48)
            
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to server at {args.server}. Is it running?")
        except Exception as e:
            print(f"Error during RAG: {e}")
        
    elif args.command == "ingest":
        # ... (keep existing ingest logic)
        if not os.path.isdir(args.source):
            print(f"Error: Source directory {args.source} does not exist.")
            return

        print(f"Syncing podcast '{args.name}' from {args.source}")
        
        print(f"Loading models (Faster Whisper, Embedding: bge-base-zh-v1.5)...")
        # Hardcoded best model for ingestion
        model_size = "large-v3"
        
        llm = None
        if args.refine:
            print(f"Initializing LLM ({args.model}) for refinement...")
            llm = get_llm_model(args.api_key, args.base_url, args.model)

        asr = get_asr_model(model_size)
        embed = get_embedding_model()
        
        indexer = IndexingService(asr, embed, DATA_DIR, llm=llm)
        collector = CollectorService(indexer)
        
        collector.sync_podcast(
            args.source,
            args.name,
            window_size=args.window_size, 
            stride=args.stride
        )

    elif args.command == "download":
        downloader = PodcastDownloader(args.dest)
        downloader.download(args.url, limit=args.limit, metadata_only=args.metadata_only)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
