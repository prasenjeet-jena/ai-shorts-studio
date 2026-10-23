"""
Scene detection and visual continuity module.
Uses FFmpeg native scene evaluation and OpenCV frame analysis to identify
camera cuts, shot boundaries, and visual stability without external heavy models.
"""

import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
from src.config import config


class SceneDetector:
    """
    Detects scene transitions, camera switches, and visual repetition
    using native FFmpeg hardware-accelerated scene metrics.
    """

    def __init__(self, ffmpeg_path: Optional[str] = None):
        self.ffmpeg_path = ffmpeg_path or config.FFMPEG_PATH

    def detect_scene_cuts(
        self,
        video_path: Path,
        start_time: float,
        duration: float,
        threshold: float = 0.28
    ) -> List[float]:
        """
        Detects exact timestamps of scene cuts / camera angle switches in seconds.
        Uses FFmpeg's fast C-level scene filter.
        """
        end_time = start_time + duration
        cmd = [
            self.ffmpeg_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-i", str(video_path),
            "-filter:v", f"select=gt(scene\\,{threshold}),metadata=print:file=-",
            "-f", "null",
            "-"
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )
            cuts: List[float] = []
            for line in proc.stdout.splitlines():
                if "pts_time:" in line:
                    match = re.search(r"pts_time:([0-9.]+)", line)
                    if match:
                        rel_t = float(match.group(1))
                        abs_t = round(start_time + rel_t, 3)
                        # Filter out duplicate detections within 0.4s
                        if not cuts or abs_t - cuts[-1] > 0.4:
                            cuts.append(abs_t)
            return cuts
        except Exception as e:
            print(f"[SceneDetector] Warning during scene detection: {e}")
            return []

    def analyze_shot_stability(
        self,
        video_path: Path,
        start_time: float,
        duration: float
    ) -> Dict[str, Any]:
        """
        Analyzes whether a clip is a single continuous shot or multi-shot,
        and measures shot lengths for pacing coordination.
        """
        cuts = self.detect_scene_cuts(video_path, start_time, duration)
        end_time = start_time + duration

        boundaries = [start_time] + cuts + [end_time]
        shots = []
        for i in range(len(boundaries) - 1):
            s_dur = round(boundaries[i + 1] - boundaries[i], 2)
            shots.append({
                "shot_index": i + 1,
                "start": boundaries[i],
                "end": boundaries[i + 1],
                "duration": s_dur
            })

        is_single_shot = len(cuts) == 0
        avg_shot_length = round(duration / (len(cuts) + 1), 2)

        return {
            "cuts": cuts,
            "cut_count": len(cuts),
            "is_single_shot": is_single_shot,
            "avg_shot_length": avg_shot_length,
            "shots": shots,
            "recommendation": "dynamic_zooms_recommended" if is_single_shot else "preserve_natural_cuts"
        }
