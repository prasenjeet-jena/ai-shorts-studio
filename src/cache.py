import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.config import config
from src.logger import default_logger

CACHE_VERSION = "2.0"


class CacheManager:
    """
    Production Cache Manager with:
    - Stable versioned cache keys.
    - Fast non-blocking file fingerprinting.
    - Safe local persistence and atomic invalidation.
    - Legacy Phase 1-5 backwards compatibility.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_fingerprint(file_path: Path) -> str:
        """
        Fast, non-blocking file fingerprint using size and last modification timestamp.
        Avoids reading gigabytes of media while reliably detecting file updates.
        """
        if not file_path.exists():
            return "missing"
        stat = file_path.stat()
        raw = f"{stat.st_size}_{int(stat.st_mtime)}"
        return hashlib.md5(raw.encode('utf-8')).hexdigest()[:12]

    def _build_cache_filename(self, video_id: str, artifact_type: str) -> Path:
        return self.cache_dir / f"{video_id}_{artifact_type}_v{CACHE_VERSION}.json"

    def save(
        self,
        video_id: str,
        artifact_type: str,
        data: Any,
        fingerprint: Optional[str] = None
    ) -> Path:
        target = self._build_cache_filename(video_id, artifact_type)
        payload = {
            "cache_version": CACHE_VERSION,
            "video_id": video_id,
            "artifact_type": artifact_type,
            "fingerprint": fingerprint or "default",
            "cached_at": time.time(),
            "data": data
        }
        # Atomic write
        tmp_target = target.with_suffix(".tmp")
        with open(tmp_target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_target.replace(target)
        default_logger.cache_event(f"{video_id}:{artifact_type}", hit=True, details=f"Saved to {target.name}")
        return target

    def load(
        self,
        video_id: str,
        artifact_type: str,
        fingerprint: Optional[str] = None
    ) -> Optional[Any]:
        target = self._build_cache_filename(video_id, artifact_type)

        # Check versioned cache first
        if target.exists():
            try:
                with open(target, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if entry.get("cache_version") != CACHE_VERSION:
                    default_logger.cache_event(f"{video_id}:{artifact_type}", hit=False, details="Version mismatch")
                    return None
                if fingerprint and entry.get("fingerprint") != "default" and entry.get("fingerprint") != fingerprint:
                    default_logger.cache_event(f"{video_id}:{artifact_type}", hit=False, details="Fingerprint mismatch")
                    return None
                default_logger.cache_event(f"{video_id}:{artifact_type}", hit=True, details="Versioned hit")
                return entry.get("data")
            except Exception as e:
                default_logger.warning(f"Error reading cache file {target}: {e}")
                return None

        # Backwards compatibility check: legacy Phase 1–5 unversioned cache files
        legacy_file = self.cache_dir / f"{video_id}_{artifact_type}.json"
        if legacy_file.exists():
            try:
                with open(legacy_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Auto-migrate legacy cache entry to v2.0
                self.save(video_id, artifact_type, data, fingerprint=fingerprint)
                default_logger.cache_event(f"{video_id}:{artifact_type}", hit=True, details="Migrated legacy hit")
                return data
            except Exception as e:
                default_logger.warning(f"Error reading legacy cache file {legacy_file}: {e}")
                return None

        default_logger.cache_event(f"{video_id}:{artifact_type}", hit=False, details="Not found")
        return None

    def has(self, video_id: str, artifact_type: str, fingerprint: Optional[str] = None) -> bool:
        return self.load(video_id, artifact_type, fingerprint=fingerprint) is not None

    def invalidate(
        self,
        video_id: Optional[str] = None,
        artifact_type: Optional[str] = None
    ) -> int:
        """
        Safely deletes matching cache files without touching source media or generated clips.
        Returns the count of invalidated files.
        """
        removed = 0
        for f in self.cache_dir.glob("*.json"):
            name = f.stem
            # Pattern: <video_id>_<artifact_type>[_v<ver>]
            if video_id and not name.startswith(video_id):
                continue
            if artifact_type and artifact_type not in name:
                continue
            try:
                f.unlink(missing_ok=True)
                removed += 1
            except Exception as e:
                default_logger.warning(f"Failed to remove cache file {f}: {e}")

        default_logger.info(f"Invalidated {removed} cache entries (video_id={video_id}, type={artifact_type}).")
        return removed


# Global cache manager instance
cache_manager = CacheManager()
