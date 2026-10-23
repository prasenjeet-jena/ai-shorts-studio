import math
import struct
import wave
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from src.config import config


class SoundEngine:
    """
    Context-Aware Sound Design Engine.
    Provides 100% royalty-free, locally synthesized audio cues for Story Remix:
    - record_scratch: Vinyl tape stop / needle scratch for comedic deadpan stops.
    - comic_pop: High-pitched cartoon bubble pop for stickers / sticker reveals.
    - dramatic_hit: Low sub-bass cinematic impact for heavy stakes or milestones.
    - comedic_sting: Cartoon spring boing / wobble for awkward reveals.
    - whoosh: Fast transition sweep for dynamic zoom bursts.
    """

    def __init__(self, audio_dir: Optional[Path] = None, sample_rate: int = 48000):
        self.audio_dir = audio_dir or (config.STORAGE_DIR / "assets" / "audio")
        self.sample_rate = sample_rate
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_assets()

    def ensure_assets(self) -> None:
        """Ensures all standard procedural sound cues exist on disk."""
        cues = {
            "record_scratch.wav": self._synth_record_scratch,
            "comic_pop.wav": self._synth_comic_pop,
            "dramatic_hit.wav": self._synth_dramatic_hit,
            "comedic_sting.wav": self._synth_comedic_sting,
            "whoosh.wav": self._synth_whoosh,
        }
        for name, synth_fn in cues.items():
            target = self.audio_dir / name
            if not target.exists() or target.stat().st_size < 1000:
                samples = synth_fn()
                self._save_wav(target, samples)

    def get_cue_path(self, cue_name: str) -> Path:
        """Returns the absolute Path for a sound cue, ensuring extension."""
        if not cue_name.endswith(".wav"):
            cue_name = f"{cue_name}.wav"
        target = self.audio_dir / cue_name
        if not target.exists():
            self.ensure_assets()
        return target.resolve()

    def _save_wav(self, target_path: Path, samples: np.ndarray) -> None:
        """Saves float32 audio samples [-1.0, 1.0] as 16-bit PCM WAV."""
        pcm = (np.clip(samples, -0.98, 0.98) * 32767).astype(np.int16)
        with wave.open(str(target_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())

    def _synth_record_scratch(self) -> np.ndarray:
        """Vinyl needle scratch / tape stop sound (0.45s)."""
        dur = 0.45
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        noise = np.random.uniform(-0.8, 0.8, len(t))
        sweep = np.sin(2 * np.pi * (750 * (1 - t / dur) ** 2 + 60) * t)
        env = np.exp(-t * 5.5) * (1 - np.exp(-t * 80))
        return (0.55 * noise + 0.45 * sweep) * env

    def _synth_comic_pop(self) -> np.ndarray:
        """Cartoon bubble pop ping (0.15s)."""
        dur = 0.15
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        f_pop = 350 + 1100 * np.exp(-t * 35)
        pop_sig = np.sin(2 * np.pi * f_pop * t) * np.exp(-t * 28)
        return pop_sig * 0.9

    def _synth_dramatic_hit(self) -> np.ndarray:
        """Sub-bass dramatic cinematic hit with soft saturation (0.75s)."""
        dur = 0.75
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        f_hit = 130 * np.exp(-t * 4.5) + 38
        hit_sig = np.sin(2 * np.pi * f_hit * t) * np.exp(-t * 3.5)
        hit_sig += 0.35 * np.sin(4 * np.pi * f_hit * t) * np.exp(-t * 5.0)
        return np.tanh(hit_sig * 1.5) * 0.9

    def _synth_comedic_sting(self) -> np.ndarray:
        """Cartoon spring boing / wobble frequency modulation (0.50s)."""
        dur = 0.50
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        mod = np.sin(2 * np.pi * 22 * t) * 70
        car = np.sin(2 * np.pi * (380 + mod) * t)
        return car * np.exp(-t * 4.5) * 0.85

    def _synth_whoosh(self) -> np.ndarray:
        """Fast dynamic camera whoosh sweep (0.35s)."""
        dur = 0.35
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        w_noise = np.random.uniform(-0.9, 0.9, len(t))
        w_env = np.sin(np.pi * t / dur) ** 2
        return w_noise * w_env * 0.8
