import argparse
import os
import sys
import requests
import time

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
    ingest_parser = subparsers.add_parser("ingest", help="Process local directory via server")
    ingest_parser.add_argument("--source", default="downloads", help="Directory path to process")
    
    # Download Command (Local)
    download_parser = subparsers.add_parser("download", help="Download podcast from URL (RSS or XiaoYuZhou)")
    download_parser.add_argument("--url", help="RSS Feed URL or XiaoYuZhou Podcast Page URL")
    download_parser.add_argument("--limit", type=int, default=0, help="Limit number of episodes to download (0 for all)")
    download_parser.add_argument("--dest", default="downloads", help="Destination directory")
    download_parser.add_argument("--metadata-only", action="store_true", help="Only download metadata (JSON), skip audio")

    args = parser.parse_args()

    if args.command == "search":
        url = f"{DEFAULT_SERVER_URL}/api/search/stream"
        payload = {
            "query": args.query,
            "podcast_name": args.name,
            "audio_id": args.id,
            "use_rag": True
        }
        
        try:
            print(f"Sending request to {url}...")
            req_start = time.perf_counter()
            resp = requests.post(url, json=payload, timeout=300, stream=True)
            req_elapsed_ms = (time.perf_counter() - req_start) * 1000
            resp.raise_for_status()
            print(f"Query first-byte latency: {req_elapsed_ms:.1f} ms")

            current_event = None
            results = []
            got_answer_header = False
            server_ttft_ms = None
            local_ttft_ms = None
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                payload_text = line[len("data:"):].strip()
                try:
                    event_payload = requests.models.complexjson.loads(payload_text)
                except Exception:
                    continue

                if current_event == "results":
                    results = event_payload.get("results", []) or []
                elif current_event == "ttft":
                    server_ttft_ms = event_payload.get("ttft_ms")
                    if server_ttft_ms is not None:
                        print(f"Server TTFT: {float(server_ttft_ms):.1f} ms")
                elif current_event == "answer":
                    delta = event_payload.get("delta", "")
                    if delta:
                        if local_ttft_ms is None:
                            local_ttft_ms = (time.perf_counter() - req_start) * 1000
                            print(f"Client TTFT: {local_ttft_ms:.1f} ms")
                        if not got_answer_header:
                            print("\n" + "="*20 + " ANSWER " + "="*20)
                            got_answer_header = True
                        print(delta, end="", flush=True)
                elif current_event == "done":
                    break

            if got_answer_header:
                print("\n" + "="*48)

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
        source = os.path.abspath(args.source)
        if not os.path.isdir(source):
            print(f"Error: Source directory {source} does not exist.")
            return
        ingest_url = f"{DEFAULT_SERVER_URL}/api/ingest"
        payload = {"source": source}

        try:
            print(f"Sending request to {ingest_url}...")
            resp = requests.post(ingest_url, json=payload, timeout=3600)
            resp.raise_for_status()
            data = resp.json()
            print("Ingest result:", data)
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to server at {DEFAULT_SERVER_URL}. Is it running?")
        except Exception as e:
            print(f"Error during ingest: {e}")

    elif args.command == "download":
        downloader = PodcastDownloader(args.dest)
        downloader.download(args.url, limit=args.limit, metadata_only=args.metadata_only)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
