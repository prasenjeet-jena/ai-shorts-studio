"""
Pacing optimization and dead-air reduction module.
Identifies unnecessary pauses and silences using transcript timestamps and audio boundaries,
preserving meaningful dramatic pauses, comedic timing, and breaths while eliminating sluggish dead air.
Remaps word timestamps to ensure 100% frame-accurate subtitle and zoom synchronization.
"""

from typing import List, Dict, Any, Tuple, Optional


class PacingOptimizer:
    """
    Intelligent pacing optimizer.
    Removes awkward dead air while preserving natural speech cadence and emotional delivery.
    """

    def __init__(self, mode: str = "conservative"):
        self.mode = mode.lower()

    def optimize_pacing(
        self,
        words_data: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float,
        mode: Optional[str] = None,
        scene_cuts: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Calculates keep segments for video/audio concatenation and remaps word timestamps.

        Modes:
        - 'off': Preserves 100% of the original timeline (no trimming).
        - 'conservative': Trims dead air > 0.85s down to 0.45s, keeping breaths, setups, and word margins.
        - 'moderate': Trims dead air > 0.65s down to 0.35s for punchy high-velocity editing.
        """
        active_mode = (mode or self.mode).lower()
        clip_words = [
            w for w in words_data
            if w.get("end", 0) >= clip_start and w.get("start", 0) <= clip_end
        ]

        orig_duration = round(clip_end - clip_start, 3)

        if active_mode in ["off", "none", "raw"] or not clip_words or len(clip_words) < 2:
            return {
                "mode": active_mode,
                "is_trimmed": False,
                "keep_segments": [(clip_start, clip_end)],
                "remapped_words": clip_words,
                "original_duration": orig_duration,
                "new_duration": orig_duration,
                "dead_air_trimmed": 0.0,
                "trim_count": 0,
                "edits": []
            }

        # Pacing thresholds
        if active_mode == "moderate":
            max_gap = 0.65
            target_gap = 0.35
            word_pad = 0.06
        else:  # conservative (default)
            max_gap = 0.85
            target_gap = 0.45
            word_pad = 0.08

        cuts = scene_cuts or []
        keep_segments: List[Tuple[float, float]] = []
        curr_seg_start = clip_start
        edits: List[Dict[str, Any]] = []

        # Check for pre-speech dead air at the beginning of the clip
        first_word_s = clip_words[0].get("start", clip_start)
        pre_speech_gap = first_word_s - clip_start
        if pre_speech_gap > 0.8:
            trim_amt = pre_speech_gap - 0.25
            curr_seg_start = round(clip_start + trim_amt, 3)
            edits.append({
                "type": "pre_speech_lead_in",
                "original_gap": round(pre_speech_gap, 2),
                "trimmed": round(trim_amt, 2),
                "preserved_gap": 0.25
            })

        for i in range(len(clip_words) - 1):
            w_curr = clip_words[i]
            w_next = clip_words[i + 1]
            gap = w_next.get("start", 0) - w_curr.get("end", 0)

            # Avoid trimming across a camera cut if it would cause a jump-cut artifact
            contains_cut = any(w_curr.get("end", 0) <= c <= w_next.get("start", 0) for c in cuts)

            if gap > max_gap and not contains_cut:
                cut_start = w_curr.get("end", 0) + word_pad
                excess = gap - target_gap
                cut_end = cut_start + excess

                if cut_end < (w_next.get("start", 0) - word_pad):
                    keep_segments.append((round(curr_seg_start, 3), round(cut_start, 3)))
                    edits.append({
                        "type": "internal_dead_air",
                        "between": f"{w_curr.get('word', '').strip()} -> {w_next.get('word', '').strip()}",
                        "original_gap": round(gap, 2),
                        "trimmed": round(cut_end - cut_start, 2),
                        "preserved_gap": round(target_gap, 2)
                    })
                    curr_seg_start = round(cut_end, 3)

        # Check for trailing silence at the end of the clip
        last_word_e = clip_words[-1].get("end", clip_end)
        post_speech_gap = clip_end - last_word_e
        final_seg_end = clip_end
        if post_speech_gap > 0.8:
            excess_trailing = post_speech_gap - 0.35
            final_seg_end = round(clip_end - excess_trailing, 3)
            edits.append({
                "type": "post_speech_trailing",
                "original_gap": round(post_speech_gap, 2),
                "trimmed": round(excess_trailing, 2),
                "preserved_gap": 0.35
            })

        # Append final segment
        keep_segments.append((round(curr_seg_start, 3), round(final_seg_end, 3)))

        # If no edits were actually made
        if len(keep_segments) == 1 and keep_segments[0] == (clip_start, clip_end):
            return {
                "mode": active_mode,
                "is_trimmed": False,
                "keep_segments": [(clip_start, clip_end)],
                "remapped_words": clip_words,
                "original_duration": orig_duration,
                "new_duration": orig_duration,
                "dead_air_trimmed": 0.0,
                "trim_count": 0,
                "edits": []
            }

        # Remap word timestamps onto the new spliced timeline
        remapped_words = []
        for w in clip_words:
            w_s = w.get("start", 0)
            w_e = w.get("end", 0)

            # Calculate total time trimmed prior to w_s
            shift_s = 0.0
            for seg_idx in range(len(keep_segments) - 1):
                gap_cut_s = keep_segments[seg_idx][1]
                gap_cut_e = keep_segments[seg_idx + 1][0]
                if w_s >= gap_cut_e:
                    shift_s += (gap_cut_e - gap_cut_s)

            # Calculate total time trimmed prior to w_e
            shift_e = 0.0
            for seg_idx in range(len(keep_segments) - 1):
                gap_cut_s = keep_segments[seg_idx][1]
                gap_cut_e = keep_segments[seg_idx + 1][0]
                if w_e >= gap_cut_e:
                    shift_e += (gap_cut_e - gap_cut_s)

            # Account for any initial clip start trim
            initial_trim = keep_segments[0][0] - clip_start if keep_segments else 0.0

            new_w = dict(w)
            new_w["start"] = round(w_s - shift_s - initial_trim, 3)
            new_w["end"] = round(w_e - shift_e - initial_trim, 3)
            remapped_words.append(new_w)

        new_duration = round(sum(e - s for s, e in keep_segments), 3)
        dead_air_trimmed = round(orig_duration - new_duration, 3)

        return {
            "mode": active_mode,
            "is_trimmed": True,
            "keep_segments": keep_segments,
            "remapped_words": remapped_words,
            "original_duration": orig_duration,
            "new_duration": new_duration,
            "dead_air_trimmed": dead_air_trimmed,
            "trim_count": len(edits),
            "edits": edits
        }
