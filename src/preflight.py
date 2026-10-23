import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

from src.config import config


class PreflightError(Exception):
    """Raised when environment or prerequisites fail preflight validation."""
    def __init__(self, message: str, failed_checks: List[Dict[str, Any]] = None):
        super().__init__(message)
        self.failed_checks = failed_checks or []


def check_ffmpeg() -> Dict[str, Any]:
    ffmpeg_bin = config.FFMPEG_PATH
    if not ffmpeg_bin or not Path(ffmpeg_bin).exists():
        return {
            "name": "FFmpeg Binary",
            "passed": False,
            "message": f"FFmpeg not found at {ffmpeg_bin}."
        }
    try:
        res = subprocess.run([ffmpeg_bin, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        passed = (res.returncode == 0)
        first_line = res.stdout.decode('utf-8', errors='ignore').split('\n')[0] if passed else "Execution failed"
        return {
            "name": "FFmpeg Binary",
            "passed": passed,
            "message": first_line if passed else "FFmpeg execution failed."
        }
    except Exception as e:
        return {
            "name": "FFmpeg Binary",
            "passed": False,
            "message": f"Error executing FFmpeg: {e}"
        }


def check_ffprobe() -> Dict[str, Any]:
    ffprobe_bin = config.FFPROBE_PATH
    if not ffprobe_bin or not Path(ffprobe_bin).exists():
        return {
            "name": "FFprobe Binary",
            "passed": False,
            "message": f"FFprobe not found at {ffprobe_bin}."
        }
    try:
        res = subprocess.run([ffprobe_bin, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        passed = (res.returncode == 0)
        first_line = res.stdout.decode('utf-8', errors='ignore').split('\n')[0] if passed else "Execution failed"
        return {
            "name": "FFprobe Binary",
            "passed": passed,
            "message": first_line if passed else "FFprobe execution failed."
        }
    except Exception as e:
        return {
            "name": "FFprobe Binary",
            "passed": False,
            "message": f"Error executing FFprobe: {e}"
        }


def check_node_runtime() -> Dict[str, Any]:
    node_bin = config.NODE_PATH
    if node_bin and Path(node_bin).exists():
        return {
            "name": "Node.js Runtime (yt-dlp support)",
            "passed": True,
            "message": f"Found Node runtime at {node_bin}."
        }
    which_node = shutil.which("node")
    if which_node:
        return {
            "name": "Node.js Runtime (yt-dlp support)",
            "passed": True,
            "message": f"Found Node in PATH: {which_node}."
        }
    return {
        "name": "Node.js Runtime (yt-dlp support)",
        "passed": False,
        "message": "Node runtime not located. YouTube signature challenges may fail on some restricted videos."
    }


def check_disk_space(target_path: Path, min_gb: float = 2.0) -> Dict[str, Any]:
    target_path.mkdir(parents=True, exist_ok=True)
    try:
        usage = shutil.disk_usage(target_path)
        free_gb = round(usage.free / (1024 ** 3), 2)
        passed = free_gb >= min_gb
        return {
            "name": "Disk Space",
            "passed": passed,
            "free_gb": free_gb,
            "required_gb": min_gb,
            "message": f"{free_gb} GB free on {target_path} (required: {min_gb} GB)."
        }
    except Exception as e:
        return {
            "name": "Disk Space",
            "passed": False,
            "message": f"Failed to inspect disk space: {e}"
        }


def check_writability(directories: List[Path]) -> Dict[str, Any]:
    unwritable = []
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
        test_file = d / ".preflight_write_test"
        try:
            with open(test_file, "w") as f:
                f.write("ok")
            test_file.unlink(missing_ok=True)
        except Exception:
            unwritable.append(str(d))

    if unwritable:
        return {
            "name": "Directory Writability",
            "passed": False,
            "message": f"The following directories are not writable: {', '.join(unwritable)}"
        }
    return {
        "name": "Directory Writability",
        "passed": True,
        "message": f"All {len(directories)} required working directories are writable."
    }


def check_python_modules() -> Dict[str, Any]:
    required = ["cv2", "faster_whisper", "yt_dlp", "tenacity"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        return {
            "name": "Python Dependencies",
            "passed": False,
            "message": f"Missing critical Python modules: {', '.join(missing)}"
        }
    return {
        "name": "Python Dependencies",
        "passed": True,
        "message": f"All critical modules ({', '.join(required)}) verified."
    }


def run_preflight(min_disk_gb: float = 2.0, require_node: bool = False) -> Dict[str, Any]:
    """
    Executes comprehensive preflight checks before any pipeline operations begin.
    Fails early if critical system requirements or dependencies are missing.
    """
    checks = [
        check_ffmpeg(),
        check_ffprobe(),
        check_python_modules(),
        check_writability([config.STORAGE_DIR, config.TEMP_DIR, config.CACHE_DIR, config.OUTPUT_DIR]),
        check_disk_space(config.STORAGE_DIR, min_gb=min_disk_gb)
    ]

    node_res = check_node_runtime()
    if require_node:
        checks.append(node_res)
    else:
        # Advisory warning if node is missing
        checks.append(node_res)

    # Critical failures: ffmpeg, ffprobe, python modules, writability, disk space
    critical_failed = [
        c for c in checks
        if not c["passed"] and c["name"] != "Node.js Runtime (yt-dlp support)"
    ]

    if require_node and not node_res["passed"]:
        critical_failed.append(node_res)

    is_ready = (len(critical_failed) == 0)

    result = {
        "ready": is_ready,
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
        "critical_failed": critical_failed
    }

    if not is_ready:
        fail_msgs = "; ".join(f"{c['name']}: {c['message']}" for c in critical_failed)
        raise PreflightError(f"Preflight check failed: {fail_msgs}", failed_checks=critical_failed)

    return result
