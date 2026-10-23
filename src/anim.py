"""
Procedural animation and comic video transformation engine.
Generates 100% royalty-free, locally rendered animated motion graphics
(radial bursts, speed lines) and comic book image transformations.
"""
import math
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from src.config import config


class AnimationEngine:
    """
    Produces procedural animated motion graphics and cel-shaded visual transformations.
    All assets are generated algorithmically on-device with zero external dependencies.
    """

    def __init__(self, asset_dir: Optional[Path] = None):
        self.asset_dir = asset_dir or Path("storage/assets/anim")
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.radial_burst_path = self.asset_dir / "radial_burst.mov"
        self.speed_lines_path = self.asset_dir / "speed_lines.mov"
        self._ensure_base_assets()

    def _ensure_base_assets(self):
        """Ensures the procedural transparent animation assets exist."""
        if not self.radial_burst_path.exists() or self.radial_burst_path.stat().st_size < 1000:
            self.generate_radial_burst(self.radial_burst_path)
        if not self.speed_lines_path.exists() or self.speed_lines_path.stat().st_size < 1000:
            self.generate_speed_lines(self.speed_lines_path)

    def generate_radial_burst(
        self,
        output_path: Path,
        width: int = 1080,
        height: int = 1920,
        duration: float = 1.0,
        fps: int = 30
    ) -> Path:
        """
        Synthesizes an animated comic radial burst that expands, peaks, and recedes.
        Encoded as a 32-bit transparent QuickTime RLE video.
        """
        total_frames = int(duration * fps)
        temp_dir = self.asset_dir / "temp_burst"
        temp_dir.mkdir(parents=True, exist_ok=True)

        cx, cy = width // 2, height // 2
        num_rays = 28

        for f in range(total_frames):
            t = f / float(total_frames)
            # Bell curve alpha envelope
            alpha_scale = math.sin(t * math.pi) if t < 1.0 else 0.0

            img = np.zeros((height, width, 4), dtype=np.uint8)
            if alpha_scale > 0.02:
                for i in range(num_rays):
                    angle = i * (2 * math.pi / num_rays) + (t * 0.12)
                    half_w = 0.045 * (1.0 - t * 0.4)
                    pts = np.array([
                        [cx, cy],
                        [int(cx + math.cos(angle - half_w) * 1600), int(cy + math.sin(angle - half_w) * 1600)],
                        [int(cx + math.cos(angle + half_w) * 1600), int(cy + math.sin(angle + half_w) * 1600)]
                    ], dtype=np.int32)
                    # Vibrant electric yellow with transparency
                    color = (255, 230, 0, int(220 * alpha_scale))
                    cv2.fillPoly(img, [pts], color)

                # Clear center circle so focal subject remains visible
                clear_radius = int(220 * t + 80)
                cv2.circle(img, (cx, cy), clear_radius, (0, 0, 0, 0), -1)

            frame_file = temp_dir / f"f_{f:03d}.png"
            cv2.imwrite(str(frame_file), img)

        # Encode transparent MOV
        cmd = [
            config.FFMPEG_PATH, "-y",
            "-framerate", str(fps),
            "-i", str(temp_dir / "f_%03d.png"),
            "-c:v", "qtrle",
            str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        # Cleanup PNG frames
        for p in temp_dir.glob("*.png"):
            p.unlink()
        try:
            temp_dir.rmdir()
        except Exception:
            pass

        return output_path

    def generate_speed_lines(
        self,
        output_path: Path,
        width: int = 1080,
        height: int = 1920,
        duration: float = 1.5,
        fps: int = 30
    ) -> Path:
        """
        Synthesizes animated horizontal manga/comic speed lines for whip-pans and fast action.
        """
        total_frames = int(duration * fps)
        temp_dir = self.asset_dir / "temp_speed"
        temp_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(42)

        for f in range(total_frames):
            t = f / float(total_frames)
            img = np.zeros((height, width, 4), dtype=np.uint8)

            if t < 0.2:
                alpha = t / 0.2
            elif t > 0.8:
                alpha = (1.0 - t) / 0.2
            else:
                alpha = 1.0

            num_lines = 40
            for _ in range(num_lines):
                y = int(np.random.randint(120, height - 200))
                length = int(np.random.randint(350, 950))
                thickness = int(np.random.randint(4, 12))
                x_start = int(np.random.randint(-200, width))
                x_end = x_start + length

                # High contrast white and cyan speed streaks
                c = (255, 255, 255, int(200 * alpha)) if np.random.rand() > 0.25 else (255, 240, 50, int(220 * alpha))
                cv2.line(img, (x_start, y), (x_end, y), c, thickness)

            frame_file = temp_dir / f"s_{f:03d}.png"
            cv2.imwrite(str(frame_file), img)

        cmd = [
            config.FFMPEG_PATH, "-y",
            "-framerate", str(fps),
            "-i", str(temp_dir / "s_%03d.png"),
            "-c:v", "qtrle",
            str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        for p in temp_dir.glob("*.png"):
            p.unlink()
        try:
            temp_dir.rmdir()
        except Exception:
            pass

        return output_path

    def transform_frame_to_comic(self, frame_bgr: np.ndarray, add_border: bool = True) -> np.ndarray:
        """
        Transforms a natural video frame into a rich, cel-shaded comic-book illustration:
        1. Bilateral surface smoothing (cleans compression noise while preserving hard edges).
        2. Adaptive thresholding for bold comic ink outlines.
        3. Color quantization (posterization) for cel-shading.
        4. Inking multiplication.
        5. Pop-art saturation and contrast enhancement.
        6. Stylized comic book panel double-border.
        """
        # 1. Bilateral surface filter
        smooth = cv2.bilateralFilter(frame_bgr, d=9, sigmaColor=85, sigmaSpace=85)

        # 2. Adaptive threshold for ink outlines
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 3. Color quantization
        div = 32
        quant = (smooth // div) * div + div // 2

        # 4. Multiply ink edges onto quantized color
        comic = cv2.bitwise_and(quant, edges_bgr)

        # 5. Saturated pop-art color grading
        hsv = cv2.cvtColor(comic, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.45, 0, 255)  # Saturation boost
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.10, 0, 255)  # Contrast pop
        comic_grade = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 6. Add comic book panel border
        if add_border:
            h, w = comic_grade.shape[:2]
            border_thick = 24
            cv2.rectangle(comic_grade, (border_thick, border_thick), (w - border_thick, h - border_thick), (0, 0, 0), 14)
            cv2.rectangle(comic_grade, (border_thick + 12, border_thick + 12), (w - border_thick - 12, h - border_thick - 12), (0, 240, 255), 8)

        return comic_grade

    def generate_halftone_texture(
        self,
        output_path: Path,
        width: int = 1080,
        height: int = 1920,
        duration: float = 2.0,
        fps: int = 30
    ) -> Path:
        """
        Synthesizes an animated subtle comic-book halftone dot overlay (QuickTime RLE MOV).
        """
        total_frames = int(duration * fps)
        temp_dir = self.asset_dir / "temp_halftone"
        temp_dir.mkdir(parents=True, exist_ok=True)

        spacing = 36
        dot_r = 7

        for f in range(total_frames):
            img = np.zeros((height, width, 4), dtype=np.uint8)
            t = f / float(total_frames)
            # Subtle drifting pattern
            offset_y = int(t * spacing)

            for y in range(-spacing, height + spacing, spacing):
                row_idx = (y + offset_y) // spacing
                offset_x = (spacing // 2) if (row_idx % 2 == 1) else 0
                for x in range(0, width + spacing, spacing):
                    cx = x + offset_x
                    cy = y + offset_y
                    if 0 <= cx < width and 0 <= cy < height:
                        cv2.circle(img, (cx, cy), dot_r, (20, 20, 25, 45), -1)

            frame_file = temp_dir / f"h_{f:03d}.png"
            cv2.imwrite(str(frame_file), img)

        cmd = [
            config.FFMPEG_PATH, "-y",
            "-framerate", str(fps),
            "-i", str(temp_dir / "h_%03d.png"),
            "-c:v", "qtrle",
            str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        for p in temp_dir.glob("*.png"):
            p.unlink()
        try:
            temp_dir.rmdir()
        except Exception:
            pass

        return output_path

    def create_comic_freeze_segment(
        self,
        frozen_frame_bgr: np.ndarray,
        duration: float,
        output_mp4: Path,
        sound_sfx_wav: Optional[Path] = None,
        reaction_overlay_mov: Optional[Path] = None
    ) -> Path:
        """
        Renders the freeze-frame sequence as a fully animated comic panel MP4:
        - Comic cel-shaded background frame with double border.
        - Animated radial burst overlaid on top.
        - Optional cartoon reaction interlude (e.g. buffering brain).
        - Record scratch audio sound cue mixed in lockstep.
        """
        styled_frame = self.transform_frame_to_comic(frozen_frame_bgr, add_border=True)
        temp_img = output_mp4.parent / f"{output_mp4.stem}_comic_frame.png"
        cv2.imwrite(str(temp_img), styled_frame)

        burst_mov = self.radial_burst_path
        has_sfx = sound_sfx_wav and sound_sfx_wav.exists()
        has_reaction = reaction_overlay_mov and reaction_overlay_mov.exists()

        cmd = [
            config.FFMPEG_PATH, "-y",
            "-loop", "1", "-t", f"{duration:.3f}", "-i", str(temp_img),
            "-i", str(burst_mov)
        ]
        next_input_idx = 2

        filter_parts = ["[1:v]scale=1080:1920[burst]"]
        overlay_chain = "[0:v][burst]overlay=0:0:shortest=1"

        if has_reaction:
            cmd.extend(["-i", str(reaction_overlay_mov)])
            rxn_idx = next_input_idx
            next_input_idx += 1
            filter_parts.append(f"[{rxn_idx}:v]scale=1080:1920[rxn]")
            filter_parts.append(f"{overlay_chain}[v_burst]; [v_burst][rxn]overlay=0:0:shortest=1[v]")
        else:
            filter_parts.append(f"{overlay_chain}[v]")

        if has_sfx:
            cmd.extend(["-i", str(sound_sfx_wav)])
            sfx_idx = next_input_idx
            next_input_idx += 1
            filter_parts.append(f"[{sfx_idx}:a]apad=pad_dur={duration:.3f},atrim=0:{duration:.3f},volume=0.95[a]")
        else:
            filter_parts.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={duration:.3f}[a]")

        cmd.extend([
            "-filter_complex", "; ".join(filter_parts),
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            str(output_mp4)
        ])

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        if temp_img.exists():
            temp_img.unlink()

        return output_mp4
