import re
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import yt_dlp

from src.config import config

YOUTUBE_REGEX = re.compile(
    r'^(https?://)?(www\.)?(youtube\.com/(watch\?v=|embed/|v/|shorts/)|youtu\.be/)([\w-]{11})([^\s]*)$'
)

def validate_youtube_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    return bool(YOUTUBE_REGEX.match(url.strip()))

def extract_video_id(url: str) -> Optional[str]:
    match = YOUTUBE_REGEX.match(url.strip())
    if match:
        return match.group(5)
    return None

class IngestionManager:
    def __init__(self):
        js_runtimes = {}
        if os.path.exists(config.NODE_PATH):
            js_runtimes = {'node': {'path': config.NODE_PATH}}

        self.base_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'js_runtimes': js_runtimes,
        }

    def fetch_metadata(self, url: str) -> Dict[str, Any]:
        if not validate_youtube_url(url):
            raise ValueError(f"Invalid YouTube URL: {url}")

        opts = dict(self.base_opts)
        opts['skip_download'] = True

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    raise ValueError("Could not extract video metadata.")
                
                if info.get('is_live'):
                    raise ValueError("Live streams are not supported. Please supply a finished video.")

                duration = float(info.get('duration') or 0.0)
                if duration <= 0:
                    raise ValueError("Unable to determine video duration.")

                return {
                    "video_id": info.get("id"),
                    "title": info.get("title", "Untitled Video"),
                    "author": info.get("uploader") or info.get("channel") or "Unknown Creator",
                    "duration": duration,
                    "thumbnail": info.get("thumbnail"),
                    "url": url,
                    "view_count": info.get("view_count", 0),
                    "description": (info.get("description") or "")[:500]
                }
        except yt_dlp.utils.DownloadError as e:
            msg = str(e)
            if "Private video" in msg:
                raise ValueError("This YouTube video is private.")
            elif "Sign in" in msg:
                raise ValueError("This video requires authentication or is age-restricted.")
            elif "Video unavailable" in msg:
                raise ValueError("This YouTube video is unavailable or has been removed.")
            else:
                raise ValueError(f"YouTube retrieval error: {msg}")

    def download(self, url: str, output_dir: Optional[Path] = None) -> Tuple[Path, Path, Dict[str, Any]]:
        """
        Downloads the source video and generates a dedicated 16kHz mono WAV audio file for Whisper.
        Returns: (video_path, audio_path, metadata)
        """
        meta = self.fetch_metadata(url)
        video_id = meta["video_id"]

        target_dir = output_dir or (config.TEMP_DIR / video_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        video_path = target_dir / "source.mp4"
        audio_path = target_dir / "audio_16k.wav"

        # Check if already cached
        if video_path.exists() and audio_path.exists():
            return video_path, audio_path, meta

        opts = dict(self.base_opts)
        opts.update({
            'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
            'outtmpl': str(target_dir / "source.%(ext)s"),
            'merge_output_format': 'mp4',
        })

        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        if not video_path.exists():
            matches = list(target_dir.glob("source.*"))
            if not matches:
                raise RuntimeError("Failed to locate downloaded video stream.")
            video_path = matches[0]

        # Extract 16kHz mono audio WAV for speech recognition
        cmd = [
            config.FFMPEG_PATH,
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction error: {res.stderr.decode('utf-8', errors='ignore')}")

        return video_path, audio_path, meta
