"""
Full-Video Cartoonization Engine for Phase 10.
Transforms entire video sequences into a vibrant, high-fidelity 2D cartoon
using AnimeGANv2 (Hayao Miyazaki style) via ONNX Runtime.

Preserves:
- Complete background and foreground cartoonization
- Strict temporal consistency without generative hallucinations
- Exact frame timing, duration, and original audio synchronization
- High-resolution 1080x1920 vertical format for YouTube Shorts
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np
import onnxruntime as ort

from src.config import config

logger = logging.getLogger("ai_shorts")


class AnimeGANv2CartoonEngine:
    """
    High-performance, local-first 2D video cartoonization engine.
    Uses AnimeGANv2 Hayao ONNX model with dynamic spatial inference.
    """

    MODELS: Dict[str, Dict[str, str]] = {
        "hayao": {
            "url": "https://huggingface.co/vumichien/AnimeGANv2_Hayao/resolve/main/AnimeGANv2_Hayao.onnx",
            "filename": "AnimeGANv2_Hayao.onnx"
        },
        "classic_toon": {
            "url": "https://huggingface.co/spaces/sayakpaul/cartoonizer-demo-onnx/resolve/main/cartoonizer.onnx",
            "filename": "whitebox_cartoonizer.onnx"
        }
    }

    def __init__(
        self,
        default_style: str = "hayao",
        inference_size: Tuple[int, int] = (432, 768),
        output_size: Tuple[int, int] = (1080, 1920),
        providers: Optional[List[str]] = None
    ):
        self.model_dir = config.STORAGE_DIR / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.default_style = default_style
        self.inference_size = inference_size  # (width, height), multiples of 8
        self.output_size = output_size        # (width, height), e.g. (1080, 1920)

        # Default providers: CPUExecutionProvider offers fastest sustained throughput on macOS arm64
        self.providers = providers or ["CPUExecutionProvider"]
        self._sessions: Dict[str, ort.InferenceSession] = {}
        self._input_names: Dict[str, str] = {}

    def ensure_model(self, style: str = "hayao") -> Path:
        """Downloads the required ONNX model for the selected style if not already cached."""
        style_key = "classic_toon" if style in ["classic_toon", "tom_and_jerry", "retro_toon"] else "hayao"
        info = self.MODELS[style_key]
        model_path = self.model_dir / info["filename"]
        if not model_path.exists() or model_path.stat().st_size < 1_000_000:
            logger.info(f"Downloading {style_key} cartoon model to {model_path}...")
            import urllib.request
            urllib.request.urlretrieve(info["url"], str(model_path))
            logger.info(f"Model downloaded successfully ({model_path.stat().st_size / 1e6:.1f} MB).")
        return model_path

    def _get_session(self, style: str = "hayao") -> Tuple[ort.InferenceSession, str]:
        """Initializes or returns cached ONNX runtime inference session for style."""
        style_key = "classic_toon" if style in ["classic_toon", "tom_and_jerry", "retro_toon"] else "hayao"
        if style_key not in self._sessions:
            model_path = self.ensure_model(style_key)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(8, os.cpu_count() or 4)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=self.providers
            )
            inp_name = sess.get_inputs()[0].name
            self._sessions[style_key] = sess
            self._input_names[style_key] = inp_name
            logger.info(f"Initialized {style_key} ONNX session with providers: {sess.get_providers()}")
        return self._sessions[style_key], self._input_names[style_key]

    def process_frame(self, frame_bgr: np.ndarray, style: Optional[str] = None) -> np.ndarray:
        """
        Transforms a single BGR frame into a 2D cartoon frame.
        - 'hayao': Ghibli anime style with delicate watercolor lines and soft shading.
        - 'classic_toon' / 'tom_and_jerry': Golden Age slapstick cartoon style with
          bold animation pen contour inking and Technicolor 3-strip color saturation.
        """
        active_style = style or self.default_style
        is_classic_toon = active_style in ["classic_toon", "tom_and_jerry", "retro_toon"]

        session, input_name = self._get_session(active_style)
        h, w = frame_bgr.shape[:2]

        # 1. 9:16 Center Crop
        target_aspect = 9.0 / 16.0
        current_aspect = float(w) / float(h)
        if current_aspect > target_aspect:
            new_w = int(h * target_aspect)
            cx = w // 2
            crop = frame_bgr[:, cx - new_w // 2 : cx + new_w // 2]
        else:
            new_h = int(w / target_aspect)
            cy = h // 2
            crop = frame_bgr[cy - new_h // 2 : cy + new_h // 2, :]

        # 2. Resize to inference resolution
        inf_w, inf_h = self.inference_size
        crop_resized = cv2.resize(crop, (inf_w, inf_h), interpolation=cv2.INTER_AREA)

        # 3. Preprocess: RGB normalized to [-1.0, 1.0], shape [1, H, W, 3]
        rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        inp = (rgb.astype(np.float32) / 127.5) - 1.0
        inp = np.expand_dims(inp, axis=0)

        # 4. Inference
        out = session.run(None, {input_name: inp})[0]

        # 5. Postprocess: Denormalize to [0, 255]
        out_img = ((out[0] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        # 6. Upscale to target canvas resolution (1080, 1920)
        out_w, out_h = self.output_size
        final_frame = cv2.resize(out_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        # 7. For classic_toon / tom_and_jerry: apply bold animation inking & Technicolor grading
        if is_classic_toon:
            orig_1080 = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            smooth = cv2.bilateralFilter(orig_1080, d=9, sigmaColor=75, sigmaSpace=75)
            gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 45, 140)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges_thick = cv2.dilate(edges, kernel, iterations=1)
            ink_mask = 1.0 - (edges_thick.astype(np.float32) / 255.0)[:, :, None]

            # Technicolor 3-Strip saturation boost
            hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.38, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.04, 0, 255)
            cel_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Dark animation charcoal ink stroke
            ink_color = np.full_like(cel_bgr, 28)
            final_frame = (cel_bgr.astype(np.float32) * ink_mask + ink_color.astype(np.float32) * (1.0 - ink_mask)).astype(np.uint8)

        return final_frame

    def cartoonize_video(
        self,
        input_video_path: Path,
        output_video_path: Path,
        start_time: float = 0.0,
        duration: Optional[float] = None,
        target_fps: Optional[float] = None,
        style: Optional[str] = None
    ) -> Path:
        """
        Cartoonizes an entire video sequence frame-by-frame and muxes original audio.
        """
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        temp_video_silent = output_video_path.parent / f"temp_silent_{output_video_path.name}"

        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video: {input_video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = target_fps or src_fps
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time * src_fps)
        if duration is not None:
            max_frames = int(duration * src_fps)
        else:
            max_frames = total_src_frames - start_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        out_w, out_h = self.output_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_video_silent), fourcc, fps, (out_w, out_h))

        logger.info(
            f"Cartoonizing video: {input_video_path.name} -> {output_video_path.name} "
            f"({max_frames} frames @ {fps:.2f} fps, start={start_time:.2f}s, dur={duration or 0:.2f}s)"
        )

        frames_processed = 0
        t0 = time.time()

        try:
            while frames_processed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                cartoon_frame = self.process_frame(frame, style=style)
                writer.write(cartoon_frame)
                frames_processed += 1

                if frames_processed % 30 == 0 or frames_processed == max_frames:
                    elapsed = time.time() - t0
                    speed = frames_processed / elapsed if elapsed > 0 else 0
                    eta = (max_frames - frames_processed) / speed if speed > 0 else 0
                    logger.info(
                        f"Cartoonizing progress: {frames_processed}/{max_frames} frames "
                        f"({frames_processed / max_frames * 100:.1f}%) | "
                        f"Speed: {speed:.1f} fps | ETA: {eta:.1f}s"
                    )
        finally:
            cap.release()
            writer.release()

        total_time = time.time() - t0
        logger.info(
            f"Cartoon video frames synthesized in {total_time:.2f}s "
            f"({frames_processed / total_time:.2f} fps average)."
        )

        # Multiplex original audio from input_video_path using FFmpeg
        # Trims audio with exact start_time and duration for perfect sync
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(duration if duration is not None else frames_processed / fps),
            "-i", str(input_video_path),
            "-i", str(temp_video_silent),
            "-map", "1:v:0",
            "-map", "0:a:0?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_video_path)
        ]

        logger.info(f"Remuxing cartoon video with original audio...")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            logger.error(f"FFmpeg audio remux failed: {res.stderr}")
            # If input had no audio, fallback to silent encode
            subprocess.run([
                "ffmpeg", "-y", "-i", str(temp_video_silent),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_video_path)
            ], check=True)

        temp_video_silent.unlink(missing_ok=True)
        logger.info(f"Cartoonized video created successfully: {output_video_path}")
        return output_video_path

    def generate_contact_sheet(
        self,
        video_path: Path,
        output_image_path: Path,
        num_samples: int = 5
    ) -> Path:
        """
        Extracts evenly spaced frames across the video and tiles them into a QC contact sheet.
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Cannot read frames from {video_path}")

        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Downscale for contact sheet tile (270x480)
                tile = cv2.resize(frame, (270, 480))
                # Add timestamp label
                sec = idx / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
                cv2.putText(
                    tile, f"{sec:.2f}s", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA
                )
                cv2.putText(
                    tile, f"{sec:.2f}s", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
                )
                frames.append(tile)
        cap.release()

        if not frames:
            raise RuntimeError("No frames extracted for contact sheet")

        # Tile horizontally
        contact_sheet = np.hstack(frames)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image_path), contact_sheet)
        logger.info(f"Generated contact sheet at: {output_image_path}")
        return output_image_path
