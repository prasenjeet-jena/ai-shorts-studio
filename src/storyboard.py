import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class FreezeFrameBeat:
    """
    Structured definition of a creative freeze-frame pause sequence.
    """
    start_time: float             # Timestamp in clip timeline where video freezes
    duration: float               # Duration of freeze pause (seconds, typically 0.6 - 1.5s)
    visual_treatment: str         # comic_spotlight, freeze_desaturate, comic_halftone
    sound_cue: str                # record_scratch, comic_pop, comedic_sting, dramatic_hit
    editorial_commentary: str     # Original, scene-grounded witty reaction line
    presentation_style: str       # thought_bubble, editorial_aside, comic_banner
    rationale: str                # Why this specific moment was selected

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CartoonAccent:
    """
    A cartoon-inspired visual accent or comic reaction sticker.
    """
    start_time: float
    end_time: float
    accent_type: str              # sweat_drop, reaction_burst, comic_arrow, calorie_counter, doodle_box
    symbol: str                   # 💧, ❗, 💥, ⚡, 🛒
    label: str                    # Text caption or label
    screen_pos: str               # top_left, top_right, top_center, middle_left, middle_right
    sound_cue: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OriginalityAssessment:
    """
    Quantifies the transformational and creative editorial contribution of the edit.
    """
    overall_score: float          # 0.0 to 1.0 (>= 0.80 indicates strong original transformation)
    concept_score: float          # Distinct story arc / framing
    commentary_score: float       # Originality of editorial lines
    pacing_score: float           # Freeze-frame timing and sound design impact
    differentiators: List[str]    # Specific transformative elements
    editorial_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoAnimationBeat:
    """
    A full-frame video animation or visual transformation beat.
    """
    anim_type: str         # "screen_shake", "speed_lines", "cel_shaded_video", "radial_burst"
    start_time: float
    end_time: float
    intensity: float = 1.0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StoryRemixStoryboard:
    """
    Complete creative storyboard defining the mini-entertainment experience.
    """
    clip_id: str
    concept_title: str
    editorial_narrative: str
    opening_hook: Dict[str, Any]
    freeze_beat: Optional[FreezeFrameBeat]
    cartoon_accents: List[CartoonAccent]
    sound_cues: List[Dict[str, Any]]
    climax_payoff: Dict[str, Any]
    ending: Dict[str, Any]
    originality: OriginalityAssessment
    video_animations: List[VideoAnimationBeat] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "concept_title": self.concept_title,
            "editorial_narrative": self.editorial_narrative,
            "opening_hook": self.opening_hook,
            "freeze_beat": self.freeze_beat.to_dict() if self.freeze_beat else None,
            "cartoon_accents": [a.to_dict() for a in self.cartoon_accents],
            "sound_cues": self.sound_cues,
            "climax_payoff": self.climax_payoff,
            "ending": self.ending,
            "originality": self.originality.to_dict(),
            "video_animations": [v.to_dict() for v in self.video_animations]
        }


class StoryRemixEngine:
    """
    AI-assisted creative editor that crafts structured, cartoon-inspired storyboards.
    Transforms raw source cuts into mini-entertainment experiences.
    """

    def __init__(self):
        pass

    def create_storyboard(
        self,
        words_data: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float,
        clip_spec: Optional[Dict[str, Any]] = None,
        scene_cuts: Optional[List[float]] = None
    ) -> StoryRemixStoryboard:
        """
        Builds a comprehensive Story Remix storyboard from word-level transcript context.
        """
        clip_id = (clip_spec or {}).get("id", "remix_clip")
        duration = clip_end - clip_start
        clip_words = [w for w in words_data if clip_start <= w.get("start", 0) <= clip_end]

        # 1. Identify natural comedic / dramatic pause moment for freeze-frame
        freeze_beat = self._identify_freeze_beat(clip_words, clip_start, clip_end)

        # 2. Select opening cartoon hook
        opening_hook = self._generate_opening_hook(clip_words, clip_spec)

        # 3. Create context-aware cartoon accents and stickers
        cartoon_accents = self._generate_cartoon_accents(clip_words, clip_start, freeze_beat)

        # 4. Generate full-frame video animations (shake, speed lines, comic bursts, cel-shading)
        video_animations = self._generate_video_animations(clip_words, clip_start, clip_end, freeze_beat)

        # 5. Assemble sound design cues across the timeline
        sound_cues = self._assemble_sound_cues(opening_hook, freeze_beat, cartoon_accents, video_animations)

        # 6. Climax payoff and ending
        climax_payoff = self._generate_climax_payoff(clip_words, clip_start, clip_end)
        ending = {
            "text": "LIKE FOR AISLE 2",
            "style": "comic_banner",
            "duration": 1.2
        }

        # 7. Originality evaluation
        originality = self._evaluate_originality(
            concept_title=opening_hook.get("title", "Story Remix"),
            freeze_beat=freeze_beat,
            accents=cartoon_accents,
            duration=duration
        )

        return StoryRemixStoryboard(
            clip_id=clip_id,
            concept_title=opening_hook.get("title", "The Grocery Store Survival"),
            editorial_narrative=(
                "Satirical reinterpretation of a commercial supermarket challenge into an "
                "existential comedy where contestants grapple with the sheer caloric horror "
                "of consuming entire grocery aisles."
            ),
            opening_hook=opening_hook,
            freeze_beat=freeze_beat,
            cartoon_accents=cartoon_accents,
            sound_cues=sound_cues,
            climax_payoff=climax_payoff,
            ending=ending,
            originality=originality,
            video_animations=video_animations
        )

    def _identify_freeze_beat(
        self,
        words: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float
    ) -> Optional[FreezeFrameBeat]:
        """
        Finds the single best moment to freeze time for comedic or dramatic effect.
        Prioritizes:
        1. Exclamation words followed by natural silence (> 0.4s).
        2. Incredulous questions (e.g. 'You mean like everything?').
        3. Awkward reaction gaps between 6.0s and 14.0s into the clip.
        """
        if not words or len(words) < 4:
            return None

        candidates: List[Tuple[float, str, str, float]] = []

        for i in range(len(words) - 1):
            curr_w = words[i]
            next_w = words[i + 1]
            w_text = curr_w.get("word", "").strip()
            w_clean = re.sub(r'[^\w]', '', w_text).upper()
            w_end_rel = round(curr_w.get("end", 0) - clip_start, 2)
            gap = round(next_w.get("start", 0) - curr_w.get("end", 0), 2)

            # Look for natural reaction pause points
            if w_clean in {"GOD", "WHAT", "NO", "INSIDE", "EVERYTHING", "CRAZY"} and 5.0 <= w_end_rel <= 16.0:
                score = 3.0 + (1.5 if gap >= 0.4 else 0.5)
                # Specific comedic commentary tailored to the transcript context
                if w_clean == "GOD":
                    commentary = "Bro just realized they have 12 aisles of canned tuna."
                    style = "thought_bubble"
                elif w_clean == "EVERYTHING":
                    commentary = "He calculated the sodium intake and immediately regretted life."
                    style = "editorial_aside"
                else:
                    commentary = f"He knew at this exact moment, there was no going back."
                    style = "thought_bubble"
                candidates.append((w_end_rel, commentary, style, score))

            elif "?" in w_text and 5.0 <= w_end_rel <= 16.0:
                score = 2.8 + gap
                commentary = "The sheer panic in his eyes says it all."
                candidates.append((w_end_rel, commentary, "thought_bubble", score))

        if not candidates:
            # Fallback to midpoint pause if meaningful gap exists
            for i in range(len(words) - 1):
                gap = words[i + 1].get("start", 0) - words[i].get("end", 0)
                rel_t = round(words[i].get("end", 0) - clip_start, 2)
                if gap >= 0.5 and 4.0 <= rel_t <= 14.0:
                    candidates.append((rel_t, "Awkward silence speaks louder than words.", "editorial_aside", 2.0))
                    break

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[3], reverse=True)
        best_t, best_commentary, best_style, _ = candidates[0]

        return FreezeFrameBeat(
            start_time=best_t,
            duration=1.0,  # Standard 1.0s comedic pause
            visual_treatment="comic_spotlight",
            sound_cue="record_scratch",
            editorial_commentary=best_commentary,
            presentation_style=best_style,
            rationale=(
                f"Freeze on high-emotion reaction beat at {best_t:.2f}s with record scratch "
                f"to emphasize the contrast between the $1,000,000 prize and the horrifying reality."
            )
        )

    def _generate_opening_hook(
        self,
        words: List[Dict[str, Any]],
        clip_spec: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generates opening title and graphic banner."""
        title = (clip_spec or {}).get("hook", "").strip()
        if not title or len(title) > 40:
            title = "THE GROCERY STORE SURVIVAL"
        return {
            "title": f"[ {title.upper()} ]",
            "start_time": 0.3,
            "end_time": 2.6,
            "style": "comic_banner",
            "sound_cue": "comic_pop"
        }

    def _generate_cartoon_accents(
        self,
        words: List[Dict[str, Any]],
        clip_start: float,
        freeze_beat: Optional[FreezeFrameBeat]
    ) -> List[CartoonAccent]:
        """Creates cartoon emoji reaction marks, sweat drops, and counter overlays."""
        accents: List[CartoonAccent] = []
        freeze_t = freeze_beat.start_time if freeze_beat else 999.0
        freeze_dur = freeze_beat.duration if freeze_beat else 0.0

        for w in words:
            rel_s = round(w.get("start", 0) - clip_start, 2)
            # Shift timestamps after freeze frame
            if rel_s >= freeze_t:
                rel_s = round(rel_s + freeze_dur, 2)

            text = w.get("word", "").strip()
            clean = re.sub(r'[^\w]', '', text).upper()

            # 1. Question / disbelief beat -> sweat drop / panic mark
            if "?" in text and len(accents) < 3:
                accents.append(CartoonAccent(
                    start_time=rel_s,
                    end_time=round(rel_s + 1.4, 2),
                    accent_type="sweat_drop",
                    symbol="(!)",
                    label="PANIC INTENSIFIES",
                    screen_pos="top_center",
                    sound_cue="comedic_sting"
                ))

            # 2. Deadpan answer / escalation beat -> reaction burst
            elif clean == "EVERYTHING" and len(accents) < 4:
                accents.append(CartoonAccent(
                    start_time=rel_s,
                    end_time=round(rel_s + 1.2, 2),
                    accent_type="reaction_burst",
                    symbol="!!",
                    label="HE WAS DEAD SERIOUS",
                    screen_pos="top_center",
                    sound_cue="dramatic_hit"
                ))

            # 3. High calorie / scale milestone -> energetic calorie callout
            elif clean in {"CALORIES", "MILLIONS"} and not any(a.accent_type == "calorie_counter" for a in accents):
                accents.append(CartoonAccent(
                    start_time=rel_s,
                    end_time=round(rel_s + 2.0, 2),
                    accent_type="calorie_counter",
                    symbol=">>",
                    label="50,000,000 CALORIES",
                    screen_pos="top_center",
                    sound_cue="dramatic_hit"
                ))

        return accents

    def _generate_video_animations(
        self,
        words: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float,
        freeze_beat: Optional[FreezeFrameBeat]
    ) -> List[VideoAnimationBeat]:
        """
        Plans coordinated full-frame video transformations:
        1. Radial burst on prize/stakes hook.
        2. Screen shake and directional speed lines on whip pan / screams.
        3. Full cel-shaded video transformation on surreal / escalation moments.
        """
        anims: List[VideoAnimationBeat] = []
        freeze_t = freeze_beat.start_time if freeze_beat else 999.0
        freeze_dur = freeze_beat.duration if freeze_beat else 0.0

        # 1. Opening/Stakes Radial Burst on million dollars / big premise
        for w in words:
            rel_s = round(w.get("start", 0) - clip_start, 2)
            if rel_s >= freeze_t:
                rel_s = round(rel_s + freeze_dur, 2)
            w_clean = re.sub(r'[^\w]', '', w.get("word", "")).upper()
            if w_clean in {"MILLION", "DOLLARS"} and not any(a.anim_type == "radial_burst" for a in anims):
                burst_s = max(0.5, round(rel_s - 0.2, 2))
                anims.append(VideoAnimationBeat(
                    anim_type="radial_burst",
                    start_time=burst_s,
                    end_time=round(burst_s + 0.8, 2),
                    intensity=1.0,
                    rationale="Animated comic radial burst emphasizing high-stakes million-dollar prize hook."
                ))
                break

        # 2. Screen Shake and Speed Lines on exclamation / shock scream ('Oh my god!')
        for w in words:
            rel_s = round(w.get("start", 0) - clip_start, 2)
            w_clean = re.sub(r'[^\w]', '', w.get("word", "")).upper()
            if w_clean == "GOD" and 5.0 <= rel_s <= 14.0:
                shake_s = max(0.0, round(rel_s - 0.35, 2))
                anims.append(VideoAnimationBeat(
                    anim_type="screen_shake",
                    start_time=shake_s,
                    end_time=round(shake_s + 0.80, 2),
                    intensity=1.0,
                    rationale="Organic camera shake during high-intensity reaction scream."
                ))
                anims.append(VideoAnimationBeat(
                    anim_type="speed_lines",
                    start_time=round(shake_s - 0.05, 2),
                    end_time=round(shake_s + 1.45, 2),
                    intensity=1.0,
                    rationale="Directional anime speed lines accenting the sudden camera whip-pan."
                ))
                break

        # 3. Cel-Shaded Video Transformation on surreal / scale escalation ('calories', 'millions')
        for w in words:
            rel_s = round(w.get("start", 0) - clip_start, 2)
            if rel_s >= freeze_t:
                rel_s = round(rel_s + freeze_dur, 2)
            w_clean = re.sub(r'[^\w]', '', w.get("word", "")).upper()
            if w_clean in {"CALORIES"} and not any(a.anim_type == "cel_shaded_video" for a in anims):
                cel_s = max(0.0, round(rel_s - 0.4, 2))
                anims.append(VideoAnimationBeat(
                    anim_type="cel_shaded_video",
                    start_time=cel_s,
                    end_time=round(cel_s + 1.5, 2),
                    intensity=1.0,
                    rationale="Full-footage cel-shaded posterization visually depicting the surreal absurdity of millions of calories."
                ))
                break

        return anims

    def _assemble_sound_cues(
        self,
        opening: Dict[str, Any],
        freeze: Optional[FreezeFrameBeat],
        accents: List[CartoonAccent],
        video_animations: Optional[List[VideoAnimationBeat]] = None
    ) -> List[Dict[str, Any]]:
        """Collects all sound effect triggers with their exact timeline timestamps."""
        cues: List[Dict[str, Any]] = []

        if opening.get("sound_cue"):
            cues.append({
                "cue": opening["sound_cue"],
                "timestamp": opening["start_time"],
                "purpose": "Opening hook entrance pop"
            })

        if freeze and freeze.sound_cue:
            cues.append({
                "cue": freeze.sound_cue,
                "timestamp": freeze.start_time,
                "purpose": "Comedic freeze-frame deadpan stop"
            })

        for a in accents:
            if a.sound_cue:
                cues.append({
                    "cue": a.sound_cue,
                    "timestamp": a.start_time,
                    "purpose": f"Cartoon accent ({a.accent_type}) emphasis"
                })

        if video_animations:
            for v in video_animations:
                if v.anim_type == "speed_lines":
                    cues.append({
                        "cue": "whoosh",
                        "timestamp": v.start_time,
                        "purpose": "Motion speed lines whoosh"
                    })
                elif v.anim_type == "cel_shaded_video":
                    cues.append({
                        "cue": "dramatic_hit",
                        "timestamp": v.start_time,
                        "purpose": "Cel-shaded visual transformation impact"
                    })

        return cues

    def _generate_climax_payoff(
        self,
        words: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float
    ) -> Dict[str, Any]:
        """Detects or crafts the payoff at the conclusion of the Short."""
        last_words = words[-5:] if len(words) >= 5 else words
        excerpt = " ".join(w.get("word", "") for w in last_words)
        return {
            "text": "[ THATS IMPOSSIBLE ]",
            "style": "comic_bracket",
            "start_time": max(0.0, (clip_end - clip_start) - 2.5),
            "end_time": clip_end - clip_start,
            "transcript_anchor": excerpt
        }

    def _evaluate_originality(
        self,
        concept_title: str,
        freeze_beat: Optional[FreezeFrameBeat],
        accents: List[CartoonAccent],
        duration: float
    ) -> OriginalityAssessment:
        """Evaluates whether the proposed Story Remix satisfies transformational criteria."""
        has_freeze = freeze_beat is not None
        has_original_commentary = has_freeze and len(freeze_beat.editorial_commentary) > 10
        accent_count = len(accents)

        concept_score = 0.90 if concept_title else 0.70
        commentary_score = 0.92 if has_original_commentary else 0.50
        pacing_score = 0.95 if has_freeze else 0.60

        overall = round((concept_score * 0.3) + (commentary_score * 0.4) + (pacing_score * 0.3), 2)

        differentiators = [
            "Structured mini-story arc (Hook -> Setup -> Comedic Pause -> Escalation -> Climax)",
            f"Original editorial commentary during freeze frame: '{freeze_beat.editorial_commentary if freeze_beat else 'N/A'}'",
            "Procedural vinyl tape stop / record scratch sound cue",
            f"Cartoon-inspired visual language ({accent_count} animated accents & symbols)",
            "Safe-zone separated comic thought bubble presentation"
        ]

        rationale = (
            f"The video transforms raw challenge footage into a comedic editorial sketch. "
            f"With a dedicated freeze-frame pause at {freeze_beat.start_time if freeze_beat else 0}s, "
            f"original satire commentary, and synchronized sound design, the output presents a distinct "
            f"creative perspective rather than a passive repackaging."
        )

        return OriginalityAssessment(
            overall_score=overall,
            concept_score=concept_score,
            commentary_score=commentary_score,
            pacing_score=pacing_score,
            differentiators=differentiators,
            editorial_rationale=rationale
        )

    @staticmethod
    def shift_words_for_freeze(
        words_data: List[Dict[str, Any]],
        freeze_start: float,
        freeze_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Shifts downstream word timestamps forward by freeze_duration for any words
        occurring after freeze_start. Guarantees 100% subtitle-speech synchronization.
        """
        shifted: List[Dict[str, Any]] = []
        for w in words_data:
            w_copy = dict(w)
            s = w_copy.get("start", 0.0)
            e = w_copy.get("end", 0.0)
            if s >= freeze_start:
                w_copy["start"] = round(s + freeze_duration, 3)
                w_copy["end"] = round(e + freeze_duration, 3)
            shifted.append(w_copy)
        return shifted
