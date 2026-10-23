import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from faster_whisper import WhisperModel

from src.config import config

class SpeechTranscriber:
    def __init__(self, model_size: Optional[str] = None):
        self.model_size = model_size or config.WHISPER_MODEL
        self._model: Optional[WhisperModel] = None

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
                download_root=str(config.CACHE_DIR / "whisper_models")
            )
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        video_id: str,
        block_duration: float = 30.0
    ) -> Dict[str, Any]:
        """
        Transcribes audio with word-level timestamps, caching both:
        1. words.json: Fine-grained word timestamps for subtitle burning.
        2. summary.json: Condensed dialogue blocks for Agent LLM reasoning.
        """
        words_file = config.CACHE_DIR / f"{video_id}_words.json"
        summary_file = config.CACHE_DIR / f"{video_id}_summary.json"

        # Check cache
        if words_file.exists() and summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                with open(words_file, "r", encoding="utf-8") as f:
                    words_data = json.load(f)
                return {
                    "video_id": video_id,
                    "cached": True,
                    "words_file": str(words_file),
                    "summary_file": str(summary_file),
                    "words": words_data,
                    "blocks": summary_data
                }
            except Exception:
                pass

        model = self._get_model()

        # Load 16kHz mono PCM audio using wave and numpy for fast, robust in-memory decode
        import wave
        import numpy as np

        with wave.open(str(audio_path), "rb") as wf:
            raw_bytes = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(raw_bytes, dtype=np.int16).flatten().astype(np.float32) / 32768.0

        segments_gen, info = model.transcribe(
            audio_array,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400)
        )

        all_words: List[Dict[str, Any]] = []
        raw_segments: List[Dict[str, Any]] = []

        for seg in segments_gen:
            words = []
            if seg.words:
                for w in seg.words:
                    word_obj = {
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "score": round(w.probability, 2)
                    }
                    words.append(word_obj)
                    all_words.append(word_obj)

            raw_segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "words": words
            })

        # Save words file
        with open(words_file, "w", encoding="utf-8") as f:
            json.dump(all_words, f, indent=2)

        # Generate aggregated semantic blocks for Agent reasoning
        blocks: List[Dict[str, Any]] = []
        curr_words = []
        curr_start = 0.0
        block_id = 1

        for seg in raw_segments:
            if not curr_words:
                curr_start = seg["start"]
            
            curr_words.append(seg["text"])
            curr_end = seg["end"]

            # If accumulated text approaches target block duration or ends on strong punctuation
            if (curr_end - curr_start >= block_duration) or (curr_end - curr_start >= 15.0 and seg["text"].endswith((".", "!", "?"))):
                full_text = " ".join(curr_words)
                blocks.append({
                    "block_id": block_id,
                    "start": curr_start,
                    "end": curr_end,
                    "duration": round(curr_end - curr_start, 2),
                    "text": full_text,
                    "word_count": len(full_text.split())
                })
                block_id += 1
                curr_words = []

        if curr_words:
            full_text = " ".join(curr_words)
            blocks.append({
                "block_id": block_id,
                "start": curr_start,
                "end": raw_segments[-1]["end"] if raw_segments else 0.0,
                "duration": round((raw_segments[-1]["end"] if raw_segments else 0.0) - curr_start, 2),
                "text": full_text,
                "word_count": len(full_text.split())
            })

        # Save summary file
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(blocks, f, indent=2)

        return {
            "video_id": video_id,
            "cached": False,
            "language": info.language,
            "words_file": str(words_file),
            "summary_file": str(summary_file),
            "total_words": len(all_words),
            "words": all_words,
            "blocks": blocks
        }
