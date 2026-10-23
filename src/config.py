from pathlib import Path
import shutil
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _find_binary(name: str, env_var: str) -> str:
    env_val = os.getenv(env_var)
    if env_val and Path(env_val).exists():
        return env_val
    which_val = shutil.which(name)
    if which_val:
        return which_val
    home = Path.home()
    fallbacks = [
        Path("/opt/homebrew/bin") / name,
        Path("/usr/local/bin") / name,
        home / ".local" / "bin" / name,
        Path("/usr/bin") / name,
    ]
    for fb in fallbacks:
        if fb.exists():
            return str(fb)
    return name

class Config:
    PROJECT_NAME: str = "Local AI Shorts Generator"
    BASE_DIR: Path = PROJECT_ROOT
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"
    TEMP_DIR: Path = PROJECT_ROOT / "storage" / "temp"
    CACHE_DIR: Path = PROJECT_ROOT / "storage" / "cache"
    OUTPUT_DIR: Path = PROJECT_ROOT / "output" / "clips"
    
    # FFmpeg / ffprobe resolution
    FFMPEG_PATH: str = _find_binary("ffmpeg", "FFMPEG_PATH")
    FFPROBE_PATH: str = _find_binary("ffprobe", "FFPROBE_PATH")
    
    # Node runtime path for yt-dlp
    NODE_PATH: str = _find_binary("node", "NODE_PATH")

    # Whisper Settings
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base.en")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float32")

config = Config()

# Ensure directories exist
config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
