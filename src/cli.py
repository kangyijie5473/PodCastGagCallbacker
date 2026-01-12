import argparse
import os
import sys


from src.models.funasr import LocalFunASR
from src.models.embedding import LocalEmbedding
from src.models.llm import OpenAILLM
from src.services.indexer import IndexingService
from src.services.searcher import SearchService
from src.services.rag import RAGService

DATA_DIR = "data_v2"

def main():
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
    os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/cert.pem"
    
    parser = argparse.ArgumentParser(description="Podcast Search Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Process Command
    proc_parser = subparsers.add_parser("process", help="Process audio file")
    proc_parser.add_argument("--audio_path", help="Path to audio file")
    proc_parser.add_argument("--id", help="Unique ID for the audio (default: filename)")
    proc_parser.add_argument("--model-size", default="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", help="FunASR model ID")
    proc_parser.add_argument("--window-size", type=int, default=20, help="Window size in segments")
    proc_parser.add_argument("--stride", type=int, default=5, help="Window stride in segments")
    
    # Search Command
    search_parser = subparsers.add_parser("search", help="Search in processed audio")
    search_parser.add_argument("--query", help="Search query")
    search_parser.add_argument("--id", help="Audio ID to search in (optional)")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    
    # RAG Command
    rag_parser = subparsers.add_parser("rag", help="Ask a question (RAG)")
    rag_parser.add_argument("--query", help="Question")
    rag_parser.add_argument("--id", help="Audio ID")
    rag_parser.add_argument("--top-k", type=int, default=5)
    rag_parser.add_argument("--api-key", help="LLM API Key")
    rag_parser.add_argument("--base-url", help="LLM Base URL")
    rag_parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM Model name")

    args = parser.parse_args()

    if args.command == "process":
        if not os.path.exists(args.audio_path):
            print(f"Error: File {args.audio_path} not found.")
            return

        audio_id = args.id or os.path.splitext(os.path.basename(args.audio_path))[0]
        
        print(f"Loading models (FunASR: {args.model_size}, Embedding: bge-small-zh-v1.5)...")
        # Initialize models
        asr = LocalFunASR(model_name=args.model_size)
        embed = LocalEmbedding()
        
        indexer = IndexingService(asr, embed, DATA_DIR)
        res = indexer.process_audio(args.audio_path, audio_id, window_size=args.window_size, stride=args.stride)
        
        print("Processing result:", res)
        print(f"Debug data saved to {os.path.join(DATA_DIR, audio_id)}")
        
    elif args.command == "search":
        print("Loading embedding model...")
        embed = LocalEmbedding()
        searcher = SearchService(embed, DATA_DIR)
        
        results = searcher.search(args.query, audio_id=args.id, top_k=args.top_k)
        
        print(f"Top results for '{args.query}':")
        for i, res in enumerate(results):
            w = res["window"]
            print(f"{i+1}. [{res['score']:.4f}] {res['audio_id']} ({w['start']:.1f}s - {w['end']:.1f}s)")
            print(f"   Text: {w['text'][:200]}...")
            print("-" * 40)

    elif args.command == "rag":
        print("Loading embedding model...")
        embed = LocalEmbedding()
        searcher = SearchService(embed, DATA_DIR)
        
        print(f"Initializing LLM ({args.model})...")
        llm = OpenAILLM()
        rag = RAGService(searcher, llm)
        
        print(f"Thinking about: {args.query}")
        answer = rag.answer(args.id, args.query, top_k=args.top_k)
        
        print("\n" + "="*20 + " ANSWER " + "="*20)
        print(answer)
        print("="*48)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
