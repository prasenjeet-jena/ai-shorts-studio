"""
Full-Screen Comedy Remix Engine & Reaction-Driven Timeline Planner.
Plans continuous, fast-paced cartoon-inspired visual transformations,
reaction inserts, and rhythmic editing across the entire Short (1-3s pacing).
"""
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

from src.reactions import ReactionLibrary, ReactionMetadata
from src.emoji_anim import AnimatedEmojiEngine, EmojiReaction


@dataclass
class ComedyBeat:
    start_time: float
    end_time: float
    beat_type: str
    visual_treatment: str
    emotion: str = "neutral"
    reaction_asset_id: Optional[str] = None
    caption_emoji: Optional[str] = None
    sound_cue: Optional[str] = None
    rationale: str = ""
    filter_expression: Optional[str] = None

    @property
    def duration(self) -> float:
        return round(self.end_time - self.start_time, 2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FullComedyTimeline:
    clip_id: str
    concept_title: str
    beats: List[ComedyBeat]
    freeze_beat: Optional[ComedyBeat]
    original_duration: float
    remix_duration: float
    pacing_stats: Dict[str, Any]
    emoji_reactions: List[EmojiReaction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "concept_title": self.concept_title,
            "beats": [b.to_dict() for b in self.beats],
            "freeze_beat": self.freeze_beat.to_dict() if self.freeze_beat else None,
            "original_duration": self.original_duration,
            "remix_duration": self.remix_duration,
            "pacing_stats": self.pacing_stats,
            "emoji_reactions": [e.to_dict() for e in self.emoji_reactions]
        }


class FullComedyRemixEngine:
    """
    Directs the Full Comedy Remix experience.
    Analyzes transcript, scene cuts, and pauses to schedule
    a continuous progression of visual transformations every 1.5-2.5 seconds.
    """

    def __init__(self):
        self.reaction_lib = ReactionLibrary()
        self.emoji_engine = AnimatedEmojiEngine()

    def plan_timeline(
        self,
        words_data: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float,
        clip_spec: Optional[Dict[str, Any]] = None,
        scene_cuts: Optional[List[float]] = None
    ) -> FullComedyTimeline:
        """
        Creates a structured, high-energy comedy timeline with visual transformations
        every 1-3 seconds, reaction inserts, and synchronized audio accents.
        """
        clip_id = (clip_spec or {}).get("id", "comedy_remix_clip")
        duration = clip_end - clip_start
        clip_words = [w for w in words_data if clip_start <= w.get("start", 0) <= clip_end]

        # 1. Identify natural pause / reaction moment for freeze + reaction interlude
        freeze_raw_time = self._find_best_freeze_point(clip_words, clip_start, clip_end)
        freeze_dur = 1.2  # 1.2s comedic pause & reaction interlude

        # 2. Build continuous comedy beats
        raw_beats = self._generate_beats(
            clip_words=clip_words,
            clip_start=clip_start,
            clip_end=clip_end,
            freeze_point=freeze_raw_time,
            freeze_duration=freeze_dur,
            scene_cuts=scene_cuts or []
        )

        freeze_beat = next((b for b in raw_beats if b.beat_type == "comic_freeze_reaction"), None)
        remix_dur = round(duration + (freeze_dur if freeze_beat else 0.0), 2)

        # 3. Plan contextual animated emoji reactions
        raw_emojis = self.emoji_engine.plan_emoji_reactions(
            words_data=clip_words,
            clip_start=clip_start,
            clip_end=clip_end,
            clip_spec=clip_spec
        )
        shifted_emojis: List[EmojiReaction] = []
        for em in raw_emojis:
            em_dict = em.to_dict()
            if freeze_beat and em_dict["start_time"] >= freeze_raw_time:
                em_dict["start_time"] = round(em_dict["start_time"] + freeze_dur, 2)
            shifted_emojis.append(EmojiReaction(**em_dict))

        # 4. Calculate pacing statistics
        total_beats = len(raw_beats)
        avg_interval = round(remix_dur / max(1, total_beats), 2)
        pacing_stats = {
            "total_visual_beats": total_beats,
            "remix_duration": remix_dur,
            "average_interval_seconds": avg_interval,
            "pacing_compliance": "PASS" if 1.0 <= avg_interval <= 3.0 else "SUBOPTIMAL"
        }

        concept_title = (clip_spec or {}).get("hook", "FULL COMEDY REMIX")
        if len(concept_title) > 35:
            concept_title = concept_title[:32] + "..."

        return FullComedyTimeline(
            clip_id=clip_id,
            concept_title=concept_title,
            beats=raw_beats,
            freeze_beat=freeze_beat,
            original_duration=duration,
            remix_duration=remix_dur,
            pacing_stats=pacing_stats,
            emoji_reactions=shifted_emojis
        )

    def _find_best_freeze_point(
        self,
        words: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float
    ) -> float:
        """Finds the optimal timestamp for the comedic reaction freeze frame."""
        for w in words:
            rel_s = round(w.get("start", 0) - clip_start, 2)
            rel_e = round(w.get("end", 0) - clip_start, 2)
            text = re.sub(r'[^\w]', '', w.get("word", "")).upper()
            if text in ["GOD", "WHAT", "WHOA", "NO", "WAIT", "BRO", "SERIOUSLY", "IMPOSSIBLE"] and 4.0 <= rel_s <= 14.0:
                return rel_e

        # Fallback: check mid-clip gap
        for i in range(len(words) - 1):
            w1_e = round(words[i].get("end", 0) - clip_start, 2)
            w2_s = round(words[i + 1].get("start", 0) - clip_start, 2)
            if 5.0 <= w1_e <= 15.0 and (w2_s - w1_e) >= 0.35:
                return w1_e

        return 8.90

    def _generate_beats(
        self,
        clip_words: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float,
        freeze_point: float,
        freeze_duration: float,
        scene_cuts: List[float]
    ) -> List[ComedyBeat]:
        """
        Dynamically schedules continuous visual transformations targeting ~1.5 - 2.5s intervals,
        driven by the actual speech, volume, emotional words, and narrative timing of THIS clip.
        """
        beats: List[ComedyBeat] = []
        duration = clip_end - clip_start
        shift = freeze_duration if freeze_point else 0.0

        # Available procedural reaction assets
        REACTION_ASSETS = [
            "reaction_buffering_brain",
            "reaction_shock_eyes",
            "reaction_facepalm",
            "reaction_awkward_silence"
        ]

        # Calculate dynamic beat segments across the clip timeline (pacing ~1.6 - 2.2s per beat)
        num_target_beats = max(10, int(round(duration / 1.8)))
        segment_dur = round(duration / num_target_beats, 2)

        curr_t = 0.0
        beat_idx = 0
        freeze_inserted = False

        while curr_t < duration:
            next_t = round(min(duration, curr_t + segment_dur), 2)
            if next_t - curr_t < 0.8 and beats:
                # Merge tiny trailing remainder into last beat
                beats[-1].end_time = round(duration + (shift if freeze_inserted else 0.0), 2)
                break

            actual_start = round(curr_t + (shift if freeze_inserted else 0.0), 2)
            actual_end = round(next_t + (shift if freeze_inserted else 0.0), 2)

            # Check if freeze point lands in this segment
            if freeze_point and not freeze_inserted and (curr_t <= freeze_point < next_t or (curr_t >= freeze_point and beat_idx >= 4)):
                f_s = round(curr_t, 2)
                f_e = round(f_s + freeze_duration, 2)
                rxn_asset = REACTION_ASSETS[beat_idx % len(REACTION_ASSETS)]
                beats.append(ComedyBeat(
                    start_time=f_s,
                    end_time=f_e,
                    beat_type="comic_freeze_reaction",
                    visual_treatment="Cel-shaded comic illustration + double ink panel border + procedural reaction interlude",
                    emotion="confusion",
                    reaction_asset_id=rxn_asset,
                    caption_emoji="🧠",
                    sound_cue="record_scratch",
                    rationale="Comedic pause beat on reaction setup."
                ))
                freeze_inserted = True
                curr_t = round(next_t, 2)
                beat_idx += 1
                continue

            # Check words inside this window
            words_in_beat = [
                w for w in clip_words
                if curr_t <= (w.get("start", 0) - clip_start) < next_t
            ]
            beat_text = " ".join(w.get("word", "") for w in words_in_beat).lower()

            # Dynamic beat type assignment based on speech & narrative role
            if beat_idx == 0:
                b_type = "comic_hook_zoom"
                b_sfx = "comic_pop"
                b_vis = "Dynamic punch-in zoom on opening hook statement"
                opener_emojis = ["🔥", "😲", "😂", "🤔", "🎉"]
                b_emoji = opener_emojis[int(clip_start) % len(opener_emojis)]
            elif any(k in beat_text for k in ["!", "shout", "what", "whoa", "omg", "holy", "no way", "scream", "god"]):
                b_type = "speed_lines_shake"
                b_sfx = "whoosh"
                b_vis = "Harmonic camera shake + anime speed lines on high energy beat"
                b_emoji = "😱"
            elif any(k in beat_text for k in ["?", "why", "how", "wait", "think", "guess", "who", "which"]):
                b_type = "comic_spotlight"
                b_sfx = None
                b_vis = "Tightening circular comic spotlight focusing on subject"
                b_emoji = "🤔"
            elif any(k in beat_text for k in ["haha", "lol", "laugh", "joke", "funny", "bro", "dead"]):
                b_type = "deadpan_snap"
                b_sfx = "comic_pop"
                b_vis = "Deadpan snap-back to vibrant saturated pop-art"
                b_emoji = "😂"
            elif any(k in beat_text for k in ["win", "winner", "prize", "money", "million", "cash", "yes", "lets go"]):
                b_type = "radial_burst"
                b_sfx = "dramatic_hit"
                b_vis = "Expanding radial burst rays centered on speaker"
                b_emoji = "🎉"
            elif any(abs(sc - curr_t) <= 0.8 for sc in (scene_cuts or [])):
                b_type = "impact_flash_reveal"
                b_sfx = "dramatic_hit"
                b_vis = "1-frame white impact flash on camera transition"
                b_emoji = "⚡"
            elif beat_idx % 3 == 1:
                b_type = "split_screen_panel"
                b_sfx = "whoosh"
                b_vis = "Dual comic panel composition"
                b_emoji = None
            else:
                b_type = "halftone_texture"
                b_sfx = None
                b_vis = "Subtle comic-book halftone dot texture shading"
                b_emoji = None

            beats.append(ComedyBeat(
                start_time=actual_start,
                end_time=actual_end,
                beat_type=b_type,
                visual_treatment=b_vis,
                emotion="excitement",
                caption_emoji=b_emoji,
                sound_cue=b_sfx,
                rationale=f"Dynamic styling on '{beat_text[:25]}...'" if beat_text else "Paced visual transformation"
            ))

            curr_t = next_t
            beat_idx += 1

        # Fallback freeze beat if freeze_point was scheduled but not placed
        if freeze_point and not freeze_inserted and len(beats) > 2:
            f_s = beats[1].end_time
            f_e = round(f_s + freeze_duration, 2)
            beats.insert(2, ComedyBeat(
                start_time=f_s,
                end_time=f_e,
                beat_type="comic_freeze_reaction",
                visual_treatment="Comedic freeze frame",
                emotion="confusion",
                reaction_asset_id="reaction_buffering_brain",
                sound_cue="record_scratch",
                rationale="Fallback freeze beat placement"
            ))

        return beats

        return beats

    def shift_words_for_timeline(
        self,
        words_data: List[Dict[str, Any]],
        freeze_start: float,
        freeze_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Shifts downstream transcript word timestamps forward by freeze_duration,
        guaranteeing that audio speech and bottom captions remain in 100% lockstep.
        """
        shifted = []
        for w in words_data:
            w_copy = dict(w)
            s = w_copy.get("start", 0)
            e = w_copy.get("end", 0)
            if s >= freeze_start:
                w_copy["start"] = round(s + freeze_duration, 3)
                w_copy["end"] = round(e + freeze_duration, 3)
            elif e > freeze_start:
                w_copy["end"] = round(e + freeze_duration, 3)
            shifted.append(w_copy)
        return shifted
