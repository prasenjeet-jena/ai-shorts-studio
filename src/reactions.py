"""
Procedural Reaction Library and Asset Engine for Full Comedy Remix.
Generates, caches, and registers 100% legally usable, locally synthesized
cartoon reactions and interludes with verified FOSS / CC0 metadata.
"""
import json
import math
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from src.config import config
from src.sound import SoundEngine


@dataclass
class ReactionMetadata:
    asset_id: str
    name: str
    category: str
    source: str = "procedural_fai"
    license: str = "CC0-1.0 (Public Domain)"
    allowed_usage: str = "Commercial & Non-Commercial (Unrestricted FOSS)"
    duration: float = 1.0
    width: int = 1080
    height: int = 1920
    fps: int = 30
    tags: List[str] = field(default_factory=list)
    has_audio: bool = True
    audio_cue: str = "comic_pop"
    attribution: str = "None required (Synthesized on-device)"
    file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReactionLibrary:
    """
    Manages and synthesizes the local cartoon reaction library.
    Zero external downloads; all assets are procedurally constructed
    from vector mathematics, OpenCV primitives, and procedural audio.
    """

    def __init__(self, asset_dir: Optional[Path] = None):
        self.asset_dir = asset_dir or Path("storage/assets/reactions")
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.asset_dir / "registry.json"
        self.sound_engine = SoundEngine()
        self.registry: Dict[str, ReactionMetadata] = {}
        self._load_or_build_registry()

    def _load_or_build_registry(self):
        """Loads registry from disk or initializes built-in procedural assets."""
        self.register_default_reactions()

    def register_default_reactions(self):
        """Registers the core set of comedic reaction assets."""
        defaults = [
            ReactionMetadata(
                asset_id="reaction_buffering_brain",
                name="Buffering Thoughts & Awkward Loading",
                category="confusion",
                duration=1.2,
                tags=["confusion", "awkward", "pause", "buffering", "thinking", "disbelief"],
                audio_cue="record_scratch"
            ),
            ReactionMetadata(
                asset_id="reaction_shock_eyes",
                name="Cartoon Shock Eyes & Jaw Drop",
                category="surprise",
                duration=1.0,
                tags=["shock", "surprise", "eyes_pop", "disbelief", "scream"],
                audio_cue="dramatic_hit"
            ),
            ReactionMetadata(
                asset_id="reaction_celebration_confetti",
                name="Celebration Confetti & Gold Stars",
                category="celebration",
                duration=1.2,
                tags=["win", "celebration", "money", "prize", "confetti"],
                audio_cue="comic_pop"
            ),
            ReactionMetadata(
                asset_id="reaction_dramatic_iris",
                name="Cartoon Spotlight Iris In/Out",
                category="anticipation",
                duration=1.0,
                tags=["anticipation", "dramatic", "spotlight", "focus", "reveal"],
                audio_cue="comedic_sting"
            ),
            ReactionMetadata(
                asset_id="reaction_facepalm",
                name="Animated Cartoon Facepalm Slap",
                category="facepalm",
                duration=1.2,
                tags=["facepalm", "disbelief", "embarrassment", "failure", "smh"],
                audio_cue="comedic_sting"
            ),
            ReactionMetadata(
                asset_id="reaction_awkward_silence",
                name="Awkward Double Blink & Ellipsis",
                category="awkwardness",
                duration=1.4,
                tags=["awkward", "silence", "pause", "blink", "crickets", "confusion"],
                audio_cue="record_scratch"
            )
        ]

        for meta in defaults:
            target_path = self.asset_dir / f"{meta.asset_id}.mov"
            try:
                meta.file_path = str(target_path.relative_to(config.BASE_DIR))
            except ValueError:
                meta.file_path = str(target_path)
            self.registry[meta.asset_id] = meta
            if not target_path.exists() or target_path.stat().st_size < 1000:
                self.synthesize_reaction(meta.asset_id)

        self._save_registry()

    def _save_registry(self):
        with open(self.registry_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self.registry.items()}, f, indent=2)

    def list_reactions(self) -> List[ReactionMetadata]:
        return list(self.registry.values())

    def get_reaction(self, asset_id: str) -> Optional[ReactionMetadata]:
        return self.registry.get(asset_id)

    def find_reaction_for_moment(self, emotion: str, tags: Optional[List[str]] = None) -> Optional[ReactionMetadata]:
        """Finds the best matching reaction asset by category and semantic tags."""
        emotion_clean = emotion.lower()
        candidates = [r for r in self.registry.values() if r.category == emotion_clean]
        if not candidates and tags:
            tag_set = {t.lower() for t in tags}
            candidates = [r for r in self.registry.values() if any(t in tag_set for t in r.tags)]
        if candidates:
            return candidates[0]
        # Fallback to buffering brain for generic comic reaction
        return self.registry.get("reaction_buffering_brain")

    def synthesize_reaction(self, asset_id: str) -> Path:
        """Synthesizes the procedural animation frames and encodes to transparent QuickTime RLE MOV."""
        meta = self.registry.get(asset_id)
        if not meta:
            raise ValueError(f"Unknown reaction asset: {asset_id}")

        out_path = Path(meta.file_path)
        if not out_path.is_absolute():
            out_path = (config.BASE_DIR / out_path).resolve()
        temp_dir = self.asset_dir / f"temp_{asset_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        if asset_id == "reaction_buffering_brain":
            self._render_buffering_brain(temp_dir, meta)
        elif asset_id == "reaction_shock_eyes":
            self._render_shock_eyes(temp_dir, meta)
        elif asset_id == "reaction_celebration_confetti":
            self._render_celebration_confetti(temp_dir, meta)
        elif asset_id == "reaction_dramatic_iris":
            self._render_dramatic_iris(temp_dir, meta)
        elif asset_id == "reaction_facepalm":
            self._render_facepalm(temp_dir, meta)
        elif asset_id == "reaction_awkward_silence":
            self._render_awkward_silence(temp_dir, meta)
        else:
            self._render_buffering_brain(temp_dir, meta)

        cmd = [
            config.FFMPEG_PATH, "-y",
            "-framerate", str(meta.fps),
            "-i", str(temp_dir / "f_%03d.png"),
            "-c:v", "qtrle",
            str(out_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        # Cleanup frame directory
        for p in temp_dir.glob("*.png"):
            p.unlink()
        try:
            temp_dir.rmdir()
        except Exception:
            pass

        return out_path

    def _render_buffering_brain(self, temp_dir: Path, meta: ReactionMetadata):
        """Renders an animated retro thinking / buffering dial with spinning dots."""
        total_frames = int(meta.duration * meta.fps)
        cx, cy = meta.width // 2, meta.height // 2 - 120
        radius = 110

        for f in range(total_frames):
            img = np.zeros((meta.height, meta.width, 4), dtype=np.uint8)
            t = f / float(total_frames)

            # Alpha envelope
            alpha = math.sin(t * math.pi) if t < 1.0 else 0.0
            if alpha > 0.05:
                # 1. Dark semi-transparent comic thought card
                card_w, card_h = 680, 420
                x1, y1 = cx - card_w // 2, cy - card_h // 2 + 50
                x2, y2 = x1 + card_w, y1 + card_h
                cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 30, int(210 * alpha)), -1)
                # Double comic border
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0, int(255 * alpha)), 8)
                cv2.rectangle(img, (x1 + 8, y1 + 8), (x2 - 8, y2 - 8), (0, 230, 255, int(255 * alpha)), 4)

                # 2. Rotating circular segments (buffering ring)
                num_dots = 10
                rot_offset = t * 4 * math.pi
                for i in range(num_dots):
                    angle = i * (2 * math.pi / num_dots) + rot_offset
                    dot_x = int(cx + math.cos(angle) * radius)
                    dot_y = int((cy - 20) + math.sin(angle) * radius)
                    dot_alpha = (i + 1) / float(num_dots) * alpha
                    # Cyan to electric yellow gradient
                    cv2.circle(img, (dot_x, dot_y), 16, (0, 255, int(255 * (1 - i / num_dots)), int(240 * dot_alpha)), -1)
                    cv2.circle(img, (dot_x, dot_y), 16, (0, 0, 0, int(200 * dot_alpha)), 3)

                # 3. Comic status text: BUFFERING THOUGHTS...
                progress = min(99, int(t * 120))
                text_title = "BUFFERING REALITY..."
                text_pct = f"{progress}%"
                cv2.putText(img, text_title, (cx - 240, cy + 140), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0, int(255 * alpha)), 5, cv2.LINE_AA)
                cv2.putText(img, text_title, (cx - 240, cy + 140), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255, int(255 * alpha)), 2, cv2.LINE_AA)
                cv2.putText(img, text_pct, (cx - 40, cy - 10), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255, int(255 * alpha)), 3, cv2.LINE_AA)

            cv2.imwrite(str(temp_dir / f"f_{f:03d}.png"), img)

    def _render_shock_eyes(self, temp_dir: Path, meta: ReactionMetadata):
        """Renders cartoon springy pop-out shock eyes with pupil dilation and exclamation marks."""
        total_frames = int(meta.duration * meta.fps)
        cx, cy = meta.width // 2, meta.height // 2 - 200

        for f in range(total_frames):
            img = np.zeros((meta.height, meta.width, 4), dtype=np.uint8)
            t = f / float(total_frames)

            # Anticipation (0-0.2) -> Pop & Overshoot (0.2-0.5) -> Wobble/Recovery (0.5-1.0)
            if t < 0.2:
                scale = 1.0 - (t / 0.2) * 0.2  # Squat
                alpha = t / 0.2
            elif t < 0.45:
                # Huge pop out!
                pop_p = (t - 0.2) / 0.25
                scale = 0.8 + math.sin(pop_p * math.pi * 0.5) * 0.7  # Up to 1.5x
                alpha = 1.0
            else:
                # Settle down
                settle_p = (t - 0.45) / 0.55
                wobble = math.sin(settle_p * math.pi * 3) * math.exp(-settle_p * 3) * 0.2
                scale = 1.0 + wobble
                alpha = max(0.0, 1.0 - settle_p * 1.5)

            if alpha > 0.05:
                eye_dist = int(120 * scale)
                eye_r = int(70 * scale)

                # Draw two huge cartoon eyeballs
                for side in (-1, 1):
                    ex = cx + side * eye_dist
                    ey = cy - int(math.sin(t * math.pi) * 40)
                    # Sclera (White)
                    cv2.circle(img, (ex, ey), eye_r, (255, 255, 255, int(255 * alpha)), -1)
                    cv2.circle(img, (ex, ey), eye_r, (0, 0, 0, int(255 * alpha)), 8)
                    # Iris (Cyan pop)
                    iris_r = int(eye_r * 0.55)
                    cv2.circle(img, (ex, ey), iris_r, (255, 200, 0, int(255 * alpha)), -1)
                    # Pupil (Black)
                    pupil_r = int(iris_r * 0.6)
                    cv2.circle(img, (ex, ey), pupil_r, (0, 0, 0, int(255 * alpha)), -1)
                    # Specular highlight
                    cv2.circle(img, (ex - int(iris_r * 0.3), ey - int(iris_r * 0.3)), int(pupil_r * 0.4), (255, 255, 255, int(255 * alpha)), -1)

                # Exclamation marks above eyes
                cv2.putText(img, "!", (cx - eye_dist, cy - eye_r - 40), cv2.FONT_HERSHEY_DUPLEX, 2.2 * scale, (0, 0, 255, int(255 * alpha)), 10, cv2.LINE_AA)
                cv2.putText(img, "!", (cx - eye_dist, cy - eye_r - 40), cv2.FONT_HERSHEY_DUPLEX, 2.2 * scale, (0, 240, 255, int(255 * alpha)), 4, cv2.LINE_AA)
                cv2.putText(img, "!", (cx + eye_dist - 20, cy - eye_r - 40), cv2.FONT_HERSHEY_DUPLEX, 2.2 * scale, (0, 0, 255, int(255 * alpha)), 10, cv2.LINE_AA)
                cv2.putText(img, "!", (cx + eye_dist - 20, cy - eye_r - 40), cv2.FONT_HERSHEY_DUPLEX, 2.2 * scale, (0, 240, 255, int(255 * alpha)), 4, cv2.LINE_AA)

            cv2.imwrite(str(temp_dir / f"f_{f:03d}.png"), img)

    def _render_celebration_confetti(self, temp_dir: Path, meta: ReactionMetadata):
        """Renders multi-colored exploding confetti ribbons and golden stars."""
        total_frames = int(meta.duration * meta.fps)
        cx, cy = meta.width // 2, meta.height // 2

        np.random.seed(999)
        num_particles = 70
        particles = []
        colors = [
            (0, 240, 255),    # Electric yellow
            (255, 50, 180),   # Neon magenta
            (50, 255, 100),   # Acid green
            (255, 180, 0),    # Cyan/Azure
            (50, 100, 255)    # Orange-red
        ]
        for _ in range(num_particles):
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(400, 1100)
            color = colors[np.random.randint(0, len(colors))]
            size = np.random.randint(12, 28)
            particles.append({
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed - 200,
                "color": color,
                "size": size,
                "rot_speed": np.random.uniform(-10, 10)
            })

        for f in range(total_frames):
            img = np.zeros((meta.height, meta.width, 4), dtype=np.uint8)
            t = f / float(total_frames)
            alpha = (1.0 - t * 0.8) if t < 1.0 else 0.0

            gravity = 650.0 * (t ** 1.5)
            for p in particles:
                px = int(cx + p["vx"] * t)
                py = int(cy + p["vy"] * t + gravity)

                if 0 <= px < meta.width and 0 <= py < meta.height:
                    rot = t * p["rot_speed"]
                    c = (*p["color"], int(240 * alpha))
                    s = p["size"]
                    pts = np.array([
                        [px + int(math.cos(rot) * s), py + int(math.sin(rot) * s)],
                        [px + int(math.cos(rot + 1.57) * s * 0.5), py + int(math.sin(rot + 1.57) * s * 0.5)],
                        [px - int(math.cos(rot) * s), py - int(math.sin(rot) * s)],
                        [px - int(math.cos(rot + 1.57) * s * 0.5), py - int(math.sin(rot + 1.57) * s * 0.5)]
                    ], dtype=np.int32)
                    cv2.fillPoly(img, [pts], c)

            cv2.imwrite(str(temp_dir / f"f_{f:03d}.png"), img)

    def _render_dramatic_iris(self, temp_dir: Path, meta: ReactionMetadata):
        """Renders classic cartoon closing/opening iris spotlight mask."""
        total_frames = int(meta.duration * meta.fps)
        cx, cy = meta.width // 2, meta.height // 2
        max_r = 1200

        for f in range(total_frames):
            img = np.zeros((meta.height, meta.width, 4), dtype=np.uint8)
            t = f / float(total_frames)

            # Close down to focus, hold, snap open
            if t < 0.4:
                curr_r = int(max_r * (1.0 - (t / 0.4) * 0.75))  # Down to 300px
            elif t < 0.7:
                curr_r = 300
            else:
                curr_r = int(300 + max_r * ((t - 0.7) / 0.3))

            # Black mask surrounding circle
            mask = np.full((meta.height, meta.width), 235, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), max(10, curr_r), 0, -1)
            img[:, :, 0] = 10
            img[:, :, 1] = 10
            img[:, :, 2] = 15
            img[:, :, 3] = mask

            # Gold ring on iris border
            cv2.circle(img, (cx, cy), max(10, curr_r), (0, 240, 255, 255), 10)

            cv2.imwrite(str(temp_dir / f"f_{f:03d}.png"), img)

    def _render_facepalm(self, temp_dir: Path, meta: ReactionMetadata):
        """Renders an animated cartoon gloved hand slapping forehead with impact stars and despair hold."""
        total_frames = int(meta.duration * meta.fps)
        cx, cy = meta.width // 2, meta.height // 2 - 150

        for f in range(total_frames):
            img = np.zeros((meta.height, meta.width, 4), dtype=np.uint8)
            t = f / float(total_frames)

            # Hand flight (0-0.22) -> Slap Impact (0.22-0.40) -> Despair hold (0.40-0.85) -> Slide down (0.85-1.0)
            if t < 0.22:
                p = t / 0.22
                hx = int(cx + 340 * (1.0 - p))
                hy = int(cy + 250 * (1.0 - p))
                alpha = min(1.0, p * 1.5)
                is_slapped = False
                droop = 0
            elif t < 0.85:
                hx = cx
                hy = cy
                alpha = 1.0
                is_slapped = True
                p_hold = (t - 0.22) / 0.63
                # Slight trembling droop
                tremble = int(math.sin(p_hold * math.pi * 8) * 4) if p_hold > 0.2 else 0
                droop = int(p_hold * 25) + tremble
            else:
                p_exit = (t - 0.85) / 0.15
                hx = cx
                hy = int(cy + p_exit * 140)
                alpha = max(0.0, 1.0 - p_exit * 1.2)
                is_slapped = False
                droop = 25

            if alpha > 0.05:
                # 1. Stylized comic head silhouette
                head_y = cy + droop
                cv2.circle(img, (cx, head_y), 130, (25, 25, 30, int(220 * alpha)), -1)
                cv2.circle(img, (cx, head_y), 130, (0, 240, 255, int(255 * alpha)), 8)

                # 2. Impact stars when hand strikes (0.22 - 0.42)
                if 0.22 <= t <= 0.42:
                    imp_p = (t - 0.22) / 0.20
                    star_r = int(140 + imp_p * 80)
                    for ray_i in range(8):
                        angle = ray_i * (math.pi / 4) + imp_p * 0.5
                        rx = int(cx + math.cos(angle) * star_r)
                        ry = int(head_y + math.sin(angle) * star_r)
                        cv2.circle(img, (rx, ry), 12, (0, 240, 255, int(255 * (1.0 - imp_p))), -1)
                        cv2.line(img, (cx, head_y), (rx, ry), (255, 255, 255, int(220 * (1.0 - imp_p))), 4)

                # 3. Cartoon gloved hand covering face
                hand_y = hy + droop
                # Palm
                cv2.circle(img, (hx, hand_y), 75, (255, 255, 255, int(255 * alpha)), -1)
                cv2.circle(img, (hx, hand_y), 75, (0, 0, 0, int(255 * alpha)), 6)
                # Fingers
                for fi, (dx, dy) in enumerate([(-45, -60), (-15, -75), (15, -75), (45, -60)]):
                    cv2.ellipse(img, (hx + dx, hand_y + dy), (16, 40), 0, 0, 360, (255, 255, 255, int(255 * alpha)), -1)
                    cv2.ellipse(img, (hx + dx, hand_y + dy), (16, 40), 0, 0, 360, (0, 0, 0, int(255 * alpha)), 5)
                # Thumb
                cv2.ellipse(img, (hx - 60, hand_y + 15), (18, 30), 45, 0, 360, (255, 255, 255, int(255 * alpha)), -1)
                cv2.ellipse(img, (hx - 60, hand_y + 15), (18, 30), 45, 0, 360, (0, 0, 0, int(255 * alpha)), 5)

                # Comic status label: FACEPALM
                if 0.25 <= t <= 0.85:
                    lbl = "FACEPALM!"
                    cv2.putText(img, lbl, (cx - 160, head_y + 200), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0, int(255 * alpha)), 8, cv2.LINE_AA)
                    cv2.putText(img, lbl, (cx - 160, head_y + 200), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 100, 255, int(255 * alpha)), 3, cv2.LINE_AA)

            cv2.imwrite(str(temp_dir / f"f_{f:03d}.png"), img)

    def _render_awkward_silence(self, temp_dir: Path, meta: ReactionMetadata):
        """Renders cartoon awkward double-blink, cold sweat drop, and pulsating ellipsis thought bubble."""
        total_frames = int(meta.duration * meta.fps)
        cx, cy = meta.width // 2, meta.height // 2 - 150

        for f in range(total_frames):
            img = np.zeros((meta.height, meta.width, 4), dtype=np.uint8)
            t = f / float(total_frames)

            # Blink phases:
            # 0.0 - 0.20: Open
            # 0.20 - 0.30: Quick Blink
            # 0.30 - 0.60: Open staring
            # 0.60 - 0.72: Second slow blink
            # 0.72 - 1.15: Wide bewildered stare
            # 1.15 - 1.40: Fade out
            if 0.20 <= t <= 0.30:
                blink = math.sin((t - 0.20) / 0.10 * math.pi)
            elif 0.60 <= t <= 0.72:
                blink = math.sin((t - 0.60) / 0.12 * math.pi)
            else:
                blink = 0.0

            alpha = 1.0 if t < 1.15 else max(0.0, 1.0 - (t - 1.15) / 0.25)

            if alpha > 0.05:
                # Two comic eyes
                eye_w = 45
                eye_h = int(65 * (1.0 - blink * 0.92))
                for side in (-1, 1):
                    ex = cx + side * 90
                    ey = cy
                    if eye_h > 8:
                        cv2.ellipse(img, (ex, ey), (eye_w, eye_h), 0, 0, 360, (255, 255, 255, int(255 * alpha)), -1)
                        cv2.ellipse(img, (ex, ey), (eye_w, eye_h), 0, 0, 360, (0, 0, 0, int(255 * alpha)), 6)
                        # Pupils
                        pupil_x = ex + int(math.sin(t * 8) * 8)
                        cv2.circle(img, (pupil_x, ey), 16, (0, 0, 0, int(255 * alpha)), -1)
                        cv2.circle(img, (pupil_x - 4, ey - 4), 5, (255, 255, 255, int(255 * alpha)), -1)
                    else:
                        # Eye shut: single curved line
                        cv2.line(img, (ex - eye_w, ey), (ex + eye_w, ey), (0, 0, 0, int(255 * alpha)), 7)

                # Cold sweat drop running down right temple
                if 0.30 <= t <= 1.20:
                    d_p = (t - 0.30) / 0.90
                    drop_x = cx + 180
                    drop_y = int(cy - 60 + d_p * 140)
                    cv2.ellipse(img, (drop_x, drop_y), (14, 22), 20, 0, 360, (255, 200, 50, int(230 * alpha)), -1)
                    cv2.ellipse(img, (drop_x, drop_y), (14, 22), 20, 0, 360, (0, 0, 0, int(230 * alpha)), 3)

                # Comic thought bubble with ellipsis "..."
                if 0.35 <= t <= 1.25:
                    bx, by = cx, cy - 180
                    # Bubble cloud
                    cv2.ellipse(img, (bx, by), (110, 55), 0, 0, 360, (255, 255, 255, int(245 * alpha)), -1)
                    cv2.ellipse(img, (bx, by), (110, 55), 0, 0, 360, (0, 0, 0, int(255 * alpha)), 5)
                    # Tail circles
                    cv2.circle(img, (bx - 40, by + 65), 14, (255, 255, 255, int(245 * alpha)), -1)
                    cv2.circle(img, (bx - 40, by + 65), 14, (0, 0, 0, int(255 * alpha)), 4)
                    cv2.circle(img, (bx - 20, by + 90), 8, (255, 255, 255, int(245 * alpha)), -1)
                    cv2.circle(img, (bx - 20, by + 90), 8, (0, 0, 0, int(255 * alpha)), 3)

                    # Pulsating dots
                    num_dots = min(3, int((t - 0.35) * 4) + 1)
                    dots_str = "." * num_dots
                    cv2.putText(img, dots_str, (bx - 35, by + 12), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0, int(255 * alpha)), 4, cv2.LINE_AA)

            cv2.imwrite(str(temp_dir / f"f_{f:03d}.png"), img)
