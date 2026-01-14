import os
import glob
from typing import List, Optional
from tqdm import tqdm
from src.services.indexer import IndexingService

class CollectorService:
    def __init__(self, indexer: IndexingService):
        self.indexer = indexer
        self.supported_extensions = {".mp3", ".wav", ".m4a", ".mp4", ".flac"}

    def _get_audio_files(self, source_dir: str) -> List[str]:
        files = []
        if not os.path.exists(source_dir):
            return []
            
        for root, _, filenames in os.walk(source_dir):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.supported_extensions:
                    files.append(os.path.join(root, filename))
        return sorted(files)

    def _generate_id(self, filename: str) -> str:
        # Generate ID: filename stem
        # e.g. "ep01.mp3" -> "ep01"
        stem = os.path.splitext(os.path.basename(filename))[0]
        # Sanitize
        safe_id = "".join([c for c in stem if c.isalnum() or c in ('-', '_')]).strip()
        return safe_id

    def _is_indexed(self, podcast_name: str, audio_id: str) -> bool:
        # Check if index directory exists
        # Path: data/{podcast_name}/{audio_id}/
        path = os.path.join(self.indexer.index_dir, podcast_name, audio_id)
        # We consider it indexed if "windows.json" exists
        return os.path.exists(os.path.join(path, "windows.json"))

    def sync_podcast(self, source_dir: str, podcast_name: str, window_size: int = 20, stride: int = 5):
        """
        Syncs a podcast directory: processes new files, skips existing ones.
        """
        print(f"Scanning source directory: {source_dir}")
        audio_files = self._get_audio_files(source_dir)
        
        if not audio_files:
            print(f"No supported audio files found in {source_dir}")
            return

        print(f"Found {len(audio_files)} audio files.")
        
        processed_count = 0
        skipped_count = 0
        
        with tqdm(total=len(audio_files), desc=f"Syncing {podcast_name}", unit="file") as pbar:
            for file_path in audio_files:
                audio_id = self._generate_id(file_path)
                
                if self._is_indexed(podcast_name, audio_id):
                    tqdm.write(f"[SKIP] {os.path.basename(file_path)} (ID: {audio_id} already exists)")
                    skipped_count += 1
                    pbar.update(1)
                    continue
                
                pbar.set_description(f"Processing {os.path.basename(file_path)}")
                # tqdm.write(f"[PROCESS] {os.path.basename(file_path)} -> ID: {audio_id}")
                
                try:
                    self.indexer.process_audio(file_path, audio_id, podcast_name=podcast_name, window_size=window_size, stride=stride)
                    processed_count += 1
                except Exception as e:
                    tqdm.write(f"[ERROR] Failed to process {file_path}: {e}")
                
                pbar.update(1)
        
        print("-" * 40)
        print(f"Sync complete for '{podcast_name}'.")
        print(f"Processed: {processed_count}")
        print(f"Skipped:   {skipped_count}")
        print("-" * 40)
