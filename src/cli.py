import argparse
import os
import sys
import time
import requests

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.downloader import PodcastDownloader

DEFAULT_SERVER_URL = "http://127.0.0.1:5473"

def main():
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
    os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
    # Use Hugging Face Mirror if needed
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    parser = argparse.ArgumentParser(description="Podcast Search Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Search Command (API)
    search_parser = subparsers.add_parser("search", help="Search via API")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--name", help="Podcast Name to filter by")
    search_parser.add_argument("--id", help="Audio ID to search in (optional)")
    
    # Ingest Command (API)
    ingest_parser = subparsers.add_parser("ingest", help="Submit a directory for server-side processing")
    ingest_parser.add_argument("--source", required=True, help="Directory path to process")
    ingest_parser.add_argument("--server", default=DEFAULT_SERVER_URL, help="API Server URL")
    
    # Download Command (Local)
    download_parser = subparsers.add_parser("download", help="Download podcast from URL (RSS or XiaoYuZhou)")
    download_parser.add_argument("--url", help="RSS Feed URL or XiaoYuZhou Podcast Page URL")
    download_parser.add_argument("--limit", type=int, default=0, help="Limit number of episodes to download (0 for all)")
    download_parser.add_argument("--dest", default="downloads", help="Destination directory")
    download_parser.add_argument("--metadata-only", action="store_true", help="Only download metadata (JSON), skip audio")

    args = parser.parse_args()

    if args.command == "search":
        url = f"{DEFAULT_SERVER_URL}/api/search"
        payload = {
            "query": args.query,
            "podcast_name": args.name,
            "audio_id": args.id,
            "use_rag": True
        }
        
        try:
            print(f"Sending request to {url}...")
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            
            answer = data.get("answer")
            if answer:
                print("\n" + "="*20 + " ANSWER " + "="*20)
                print(answer)
                print("="*48)
            
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
            print(f"Error: Could not connect to server at {DEFAULT_SERVER_URL}. Is it running?")
        except Exception as e:
            print(f"Error during search: {e}")

    elif args.command == "ingest":
        if not os.path.isdir(args.source):
            print(f"Error: Source directory {args.source} does not exist.")
            return

        supported_extensions = {".mp3", ".wav", ".m4a", ".mp4", ".flac"}
        audio_files = []
        for root, _, filenames in os.walk(args.source):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported_extensions:
                    audio_files.append(os.path.join(root, filename))

        if not audio_files:
            print(f"No supported audio files found in {args.source}")
            return

        upload_url = f"{args.server}/api/upload"
        tasks_url = f"{args.server}/api/tasks"
        submitted = []

        try:
            print(f"Found {len(audio_files)} audio file(s), uploading to {upload_url}...")
            for audio_path in audio_files:
                with open(audio_path, "rb") as f:
                    files = {"file": (os.path.basename(audio_path), f)}
                    resp = requests.post(upload_url, files=files, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    task_id = data.get("task_id")
                    submitted.append((audio_path, task_id))
                    print(f"Uploaded: {audio_path} -> task {task_id}")

            for audio_path, task_id in submitted:
                if not task_id:
                    print(f"Task not returned for {audio_path}")
                    continue
                status = "pending"
                while status in {"pending", "processing"}:
                    resp = requests.get(f"{tasks_url}/{task_id}", timeout=30)
                    resp.raise_for_status()
                    task = resp.json()
                    status = task.get("status", "unknown")
                    progress = task.get("progress", 0)
                    message = task.get("message", "")
                    print(f"[{status}] {os.path.basename(audio_path)} {progress}% {message}")
                    if status in {"pending", "processing"}:
                        time.sleep(60)

        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to server at {args.server}. Is it running?")
        except Exception as e:
            print(f"Error during ingest: {e}")

    elif args.command == "download":
        downloader = PodcastDownloader(args.dest)
        downloader.download(args.url, limit=args.limit, metadata_only=args.metadata_only)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
