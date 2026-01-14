import os
import requests
import feedparser
from tqdm import tqdm
import re
import json
import time
from datetime import datetime
from typing import Optional, List, Dict

class PodcastDownloader:
    def __init__(self, download_dir: str):
        self.download_dir = download_dir
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

    def download(self, url: str, limit: int = 0):
        if "xiaoyuzhoufm.com" in url:
            self._download_xiaoyuzhou(url, limit)
        else:
            self._download_rss(url, limit)

    def _download_xiaoyuzhou(self, url: str, limit: int):
        print(f"Detected XiaoYuZhou URL: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            
            match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', resp.text)
            if not match:
                print("Error: Could not find podcast data on XiaoYuZhou page.")
                return

            data = json.loads(match.group(1))
            props = data.get('props', {}).get('pageProps', {})
            podcast = props.get('podcast', {})
            
            title = podcast.get('title', 'Unknown_Podcast')
            episodes = podcast.get('episodes', [])
            
            print(f"Podcast Title: {title}")
            print(f"Found {len(episodes)} episodes.")
            
            if limit > 0:
                episodes = episodes[:limit]
                print(f"Limiting to latest {limit} episodes.")

            self._process_episodes(title, episodes, is_xyz=True)
            
        except Exception as e:
            print(f"Error downloading from XiaoYuZhou: {e}")

    def _download_rss(self, url: str, limit: int):
        print(f"Parsing RSS Feed: {url}")
        try:
            feed = feedparser.parse(url, agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        except Exception as e:
            print(f"Error fetching feed: {e}")
            return

        if feed.bozo:
            print(f"Warning: Feed parsing had issues: {feed.bozo_exception}")
            
        if not feed.entries:
            print("No entries found in feed.")
            return

        podcast_title = feed.feed.get("title", "Unknown_Podcast")
        print(f"Podcast Title: {podcast_title}")
        
        entries = feed.entries
        if limit > 0:
            entries = entries[:limit]
            print(f"Limiting to latest {limit} episodes.")
            
        self._process_episodes(podcast_title, entries, is_xyz=False)

    def _process_episodes(self, podcast_title: str, episodes: List[any], is_xyz: bool):
        # Create safe directory name
        safe_title = "".join([c for c in podcast_title if c.isalnum() or c in (' ', '-', '_')]).strip()
        safe_title = safe_title.replace(" ", "_")
        podcast_dir = os.path.join(self.download_dir, safe_title)
        os.makedirs(podcast_dir, exist_ok=True)
        
        print(f"Processing {len(episodes)} episodes...")
        success_count = 0
        
        for entry in episodes:
            if is_xyz:
                title = entry.get("title", "Untitled")
                audio_url = entry.get("enclosure", {}).get("url")
                pub_date_str = entry.get("pubDate") # ISO string likely
                # XYZ pubDate is usually ISO 8601
                date_str = "00000000"
                if pub_date_str:
                    try:
                        dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                        date_str = dt.strftime("%Y%m%d")
                    except:
                        pass
            else:
                title = entry.get("title", "Untitled")
                audio_url = None
                for enc in entry.get("enclosures", []):
                    if enc.get("type", "").startswith("audio"):
                        audio_url = enc.get("href")
                        break
                if not audio_url:
                    for link in entry.get("links", []):
                        if link.get("type", "").startswith("audio"):
                            audio_url = link.get("href")
                            break
                
                published = entry.get("published_parsed")
                date_str = "00000000"
                if published:
                    date_str = f"{published.tm_year:04d}{published.tm_mon:02d}{published.tm_mday:02d}"

            if not audio_url:
                print(f"[SKIP] '{title}': No audio source found.")
                continue

            safe_ep_title = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_')]).strip()
            safe_ep_title = safe_ep_title[:50]
            
            # Determine extension
            ext = ".mp3"
            if ".m4a" in audio_url:
                ext = ".m4a"
            elif ".wav" in audio_url:
                ext = ".wav"
                
            filename = f"{date_str}_{safe_ep_title}{ext}"
            filepath = os.path.join(podcast_dir, filename)
            
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                print(f"[SKIP] '{title}': File exists.")
                continue
            
            print(f"[DOWNLOADING] '{title}'...")
            if self._download_file(audio_url, filepath):
                success_count += 1
                
        print("-" * 40)
        print(f"Download complete. Downloaded: {success_count} files.")
        print(f"Saved to: {podcast_dir}")
        print("-" * 40)

    def _download_file(self, url: str, filepath: str) -> bool:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, stream=True, timeout=60, headers=headers)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 
            
            with open(filepath, 'wb') as file, tqdm(
                desc="Progress",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                leave=False
            ) as bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    bar.update(size)
            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
