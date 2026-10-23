import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

class SmartFramingEngine:
    """
    Intelligent 9:16 vertical reframing engine.
    Detects the primary subject/face in horizontal footage and creates a smooth,
    centered vertical crop without black bars or image distortion.
    Falls back gracefully to a blurred background composite for wide scenes or when no face is found.
    """

    def __init__(self):
        cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

    def analyze_clip_framing(
        self,
        video_path: Path,
        start_time: float,
        duration: float,
        framing_mode: str = "auto"  # "auto", "crop", "blur"
    ) -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                "layout": "blur_composite",
                "reason": "Could not open video stream for analysis",
                "crop_x": 0
            }

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # If source is already vertical (height >= width), pass through
        if h >= w:
            cap.release()
            return {
                "layout": "vertical_passthrough",
                "width": w,
                "height": h,
                "reason": "Source video is already vertical"
            }

        if framing_mode == "blur":
            cap.release()
            return {
                "layout": "blur_composite",
                "width": w,
                "height": h,
                "reason": "Explicit blur framing requested"
            }

        # Sample 16 frames evenly across the clip duration
        num_samples = 16
        sample_times = np.linspace(start_time, start_time + duration, num=num_samples)
        face_x_centers = []
        face_widths = []

        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(int(h * 0.08), int(h * 0.08))
            )

            if len(faces) > 0:
                # If multiple faces spread wide across the screen, flag wide shot
                if len(faces) >= 2:
                    xs = [f[0] + f[2]//2 for f in faces]
                    spread = max(xs) - min(xs)
                    if spread > int(w * 0.35):
                        # Multi-person wide shot: cropping one would cut off the other
                        continue

                # Select largest face
                largest = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest
                cx = fx + fw // 2
                face_x_centers.append(cx)
                face_widths.append(fw)

        cap.release()

        # Target 9:16 crop window from the original frame
        # e.g., for 1920x1080 source, crop width = 1080 * 9 / 16 = 608px
        crop_w = int(h * 9 / 16)
        # Ensure crop width is even for H.264
        if crop_w % 2 != 0:
            crop_w += 1
        crop_h = h

        detection_rate = len(face_x_centers) / float(num_samples)

        # If a consistent face is detected in >= 30% of frames (or forced crop)
        if (detection_rate >= 0.30 and len(face_x_centers) >= 4) or framing_mode == "crop":
            # Use robust median center to eliminate outliers/jitter
            median_x = int(np.median(face_x_centers)) if face_x_centers else w // 2
            
            # Center the crop box on the median face position
            crop_x = int(median_x - (crop_w / 2))
            # Clamp to frame boundaries
            crop_x = max(0, min(w - crop_w, crop_x))
            if crop_x % 2 != 0:
                crop_x -= 1

            return {
                "layout": "smart_crop",
                "crop_x": crop_x,
                "crop_y": 0,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "face_detection_rate": round(detection_rate, 2),
                "median_face_x": median_x,
                "reason": f"Centered on speaker's face (detected in {int(detection_rate*100)}% of samples)"
            }

        # Fallback to blurred background composite
        return {
            "layout": "blur_composite",
            "crop_w": crop_w,
            "crop_h": crop_h,
            "face_detection_rate": round(detection_rate, 2),
            "reason": f"Wide action or low face presence ({int(detection_rate*100)}%); using blurred-background layout"
        }

    def build_filtergraph(
        self,
        framing_info: Dict[str, Any],
        ass_path: Path,
        zoom_intervals: Optional[List[Tuple[float, float]]] = None,
        zoom_factor: float = 1.08,
        input_v_label: str = "0:v",
        extra_filters: Optional[List[str]] = None
    ) -> str:
        """
        Builds the FFmpeg filtergraph based on framing analysis, punch-in zoom emphasis, and subtitle burning.
        """
        escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
        layout = framing_info.get("layout", "blur_composite")
        extra_str = ("," + ",".join(extra_filters)) if extra_filters else ""

        if layout == "smart_crop":
            cx = framing_info["crop_x"]
            cw = framing_info["crop_w"]
            ch = framing_info["crop_h"]

            if zoom_intervals:
                # Dynamic punch-in zoom during key emphasis statements
                cw_zoom = int(round(cw / zoom_factor))
                if cw_zoom % 2 != 0:
                    cw_zoom -= 1
                ch_zoom = int(round(ch / zoom_factor))
                if ch_zoom % 2 != 0:
                    ch_zoom -= 1
                cx_zoom = int(round(cx + (cw - cw_zoom) / 2))
                cy_zoom = int(round((ch - ch_zoom) * 0.25))

                cond_parts = [f"between(t\\, {s:.2f}\\, {e:.2f})" for s, e in zoom_intervals]
                cond = "+".join(cond_parts)
                w_expr = f"'if({cond}\\, {cw_zoom}\\, {cw})'"
                h_expr = f"'if({cond}\\, {ch_zoom}\\, {ch})'"
                x_expr = f"'if({cond}\\, {cx_zoom}\\, {cx})'"
                y_expr = f"'if({cond}\\, {cy_zoom}\\, 0)'"

                return (
                    f"[{input_v_label}]crop=w={w_expr}:h={h_expr}:x={x_expr}:y={y_expr}:exact=1,"
                    f"scale=1080:1920:flags=lanczos{extra_str},"
                    f"ass='{escaped_ass}'[v]"
                )
            else:
                return (
                    f"[{input_v_label}]crop={cw}:{ch}:{cx}:0,"
                    f"scale=1080:1920:flags=lanczos{extra_str},"
                    f"ass='{escaped_ass}'[v]"
                )
        elif layout == "vertical_passthrough":
            # Already vertical: scale to fit 1080x1920 and center crop
            return (
                f"[{input_v_label}]scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920{extra_str},"
                f"ass='{escaped_ass}'[v]"
            )
        else:
            # Blur composite fallback for wide action
            return (
                f"[{input_v_label}]split=2[bg_in][fg_in];"
                f"[bg_in]scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920,"
                f"boxblur=20:5[bg];"
                f"[fg_in]scale=1080:-2[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2{extra_str}[combined];"
                f"[combined]ass='{escaped_ass}'[v]"
            )
