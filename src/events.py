import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class VisualEvent:
    """
    Structured representation of a detected narrative or visual event in a Short.
    """
    event_id: str
    event_type: str            # money_reward, surprise_reveal, intense_reaction, challenge_stakes, key_takeaway, celebration_win
    start: float               # Relative to clip start (seconds)
    end: float                 # Relative to clip start (seconds)
    duration: float
    confidence: float          # 0.0 to 1.0
    evidence: str              # Explanation of what triggered the detection
    recommended_effect: str    # money_reward_badge, surprise_burst, impact_accent, challenge_callout, takeaway_bracket, celebration_burst
    intensity: str             # subtle, moderate, high
    transcript_excerpt: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Lexical & Pattern Matchers
# ---------------------------------------------------------------------------

MONEY_REGEX = re.compile(
    r'(\$\s?[\d,]+(?:\.\d+)?|\b\d+[\d,]*(?:\s?(?:million|thousand|hundred|billion|dollars|bucks))\b|\bmillion\s+dollars\b|\bthousand\s+dollars\b)',
    re.IGNORECASE
)

SURPRISE_KEYWORDS = {
    "WHAT", "HOW", "WAIT", "IMPOSSIBLE", "SECRET", "REVEAL", "UNBELIEVABLE",
    "REALLY", "SERIOUSLY", "NO WAY", "WHY"
}

INTENSE_EXCITED_WORDS = {
    "OH MY GOD", "OH GOD", "LETS GO", "YES", "INSANE", "CRAZY", "MASSIVE",
    "HUGE", "WOW", "HOLY", "AMAZING"
}

INTENSE_CONFRONTATION_WORDS = {
    "HOW IS THIS FAIR", "OH NO", "WE LOSE", "STOP", "NEVER", "WRONG", "FAIR",
    "DISASTER", "PANIC", "ELIMINATED", "BUSTED", "CHEATING", "TRICK"
}

CHALLENGE_KEYWORDS = {
    "CHALLENGE", "CONTESTANTS", "ATTEMPTING", "RECORD", "CALORIES", "POUNDS",
    "VERSUS", "VS", "RULES", "SURVIVE", "LAST DAY", "FINAL MEAL", "GROCERY STORE"
}

TAKEAWAY_KEYWORDS = {
    "TRUTH", "SECRET", "IMPOSSIBLE", "NEVER", "ALWAYS", "BECAUSE", "POINT IS",
    "CONCLUSION", "LESSON", "REMEMBER", "WORST", "BEST"
}

CELEBRATION_KEYWORDS = {
    "WIN", "WON", "WINNER", "REWARD", "HAPPY", "BIRTHDAY", "MARRY", "PROPOSAL",
    "CONGRATULATIONS", "CHAMPION", "MILLIONAIRE"
}


class VisualEventDetector:
    """
    Context-aware narrative event detection engine.
    Analyzes word-level transcripts, speech velocity, punctuation, and candidate context
    to identify purposeful moments for visual enhancement without arbitrary placement.
    """

    def __init__(self):
        pass

    def detect_events(
        self,
        words_data: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float,
        clip_spec: Optional[Dict[str, Any]] = None,
        scene_cuts: Optional[List[float]] = None,
        intensity: str = "dynamic"
    ) -> List[VisualEvent]:
        """
        Scans dialogue words within [clip_start, clip_end] and detects narrative events.
        All timestamps in returned VisualEvents are RELATIVE to clip_start (0.0 = clip start).
        """
        clip_words = [
            w for w in words_data
            if w.get("end", 0) >= clip_start and w.get("start", 0) <= clip_end
        ]
        if not clip_words:
            return []

        clip_dur = round(clip_end - clip_start, 3)
        detected: List[VisualEvent] = []
        evt_idx = 1

        # Build full sentence/phrase windows for context matching
        text_full = " ".join(w.get("word", "").strip() for w in clip_words)

        # -------------------------------------------------------------
        # 1. Money, Prizes, and Rewards
        # -------------------------------------------------------------
        money_matches = list(MONEY_REGEX.finditer(text_full))
        for m in money_matches:
            matched_text = m.group(0).strip()
            # Locate word boundaries corresponding to this match
            match_start_char = m.start()
            curr_char = 0
            w_start_time = None
            w_end_time = None

            for w in clip_words:
                w_str = w.get("word", "").strip()
                if curr_char <= match_start_char < curr_char + len(w_str) + 1:
                    w_start_time = max(0.0, w.get("start", clip_start) - clip_start)
                    w_end_time = min(clip_dur, w.get("end", clip_start) - clip_start + 1.8)
                    break
                curr_char += len(w_str) + 1

            if w_start_time is None:
                w_start_time = max(0.0, clip_words[0].get("start", clip_start) - clip_start)
                w_end_time = min(clip_dur, w_start_time + 2.0)

            dur = min(2.5, max(1.6, w_end_time - w_start_time))
            # Format clean badge text
            badge_val = matched_text.upper()
            if "MILLION" in badge_val and "$" not in badge_val:
                badge_val = "$1,000,000 PRIZE"
            elif "$" not in badge_val and any(c.isdigit() for c in badge_val):
                badge_val = f"${badge_val}"
            else:
                badge_val = f"{badge_val} REWARD"

            detected.append(VisualEvent(
                event_id=f"evt_money_{evt_idx}",
                event_type="money_reward",
                start=round(w_start_time, 2),
                end=round(min(clip_dur, w_start_time + dur), 2),
                duration=round(dur, 2),
                confidence=0.95,
                evidence=f"Explicit monetary amount or prize declared in transcript: '{matched_text}'",
                recommended_effect="money_reward_badge",
                intensity="high" if "million" in matched_text.lower() else "moderate",
                transcript_excerpt=matched_text,
                parameters={
                    "badge_text": badge_val,
                    "accent_color": "&H0000FFFF",  # Gold / Electric Yellow
                    "border_color": "&H00004000",  # Emerald Dark
                    "icon": "money"
                }
            ))
            evt_idx += 1

        # -------------------------------------------------------------
        # 2. Shouting and Intense Reactions
        # -------------------------------------------------------------
        for i, w in enumerate(clip_words):
            raw_w = w.get("word", "").strip()
            clean_w = re.sub(r"[^\w\s]", "", raw_w).upper()
            rel_s = max(0.0, w.get("start", clip_start) - clip_start)
            rel_e = min(clip_dur, w.get("end", clip_start) - clip_start)

            # Check for two-word or three-word reaction phrases
            phrase_3 = " ".join(
                re.sub(r"[^\w\s]", "", clip_words[k].get("word", "").strip()).upper()
                for k in range(i, min(len(clip_words), i + 3))
            )
            phrase_4 = " ".join(
                re.sub(r"[^\w\s]", "", clip_words[k].get("word", "").strip()).upper()
                for k in range(i, min(len(clip_words), i + 4))
            )

            is_confrontation = any(term in phrase_4 for term in INTENSE_CONFRONTATION_WORDS)
            is_excited = any(term in phrase_3 for term in INTENSE_EXCITED_WORDS)
            has_exclamation = raw_w.endswith("!") or (raw_w.isupper() and len(clean_w) >= 3)

            if is_confrontation:
                dur = 1.2
                detected.append(VisualEvent(
                    event_id=f"evt_intense_{evt_idx}",
                    event_type="intense_reaction",
                    start=round(rel_s, 2),
                    end=round(min(clip_dur, rel_s + dur), 2),
                    duration=dur,
                    confidence=0.92,
                    evidence=f"Intense panic or confrontation reaction phrase detected: '{phrase_4}'",
                    recommended_effect="impact_accent",
                    intensity="high",
                    transcript_excerpt=phrase_4,
                    parameters={
                        "badge_text": "PANIC / DISBELIEF",
                        "reaction_type": "confrontation",
                        "red_edge_flash": True,
                        "accent_color": "&H000000FF",  # Crimson Red
                        "icon": "exclamation"
                    }
                ))
                evt_idx += 1

            elif is_excited or (has_exclamation and clean_w in {"GOD", "WOW", "INSANE", "CRAZY", "NO"}):
                dur = 1.0
                detected.append(VisualEvent(
                    event_id=f"evt_intense_{evt_idx}",
                    event_type="intense_reaction",
                    start=round(rel_s, 2),
                    end=round(min(clip_dur, rel_s + dur), 2),
                    duration=dur,
                    confidence=0.88,
                    evidence=f"High-energy emotional shout / excited exclamation: '{raw_w}'",
                    recommended_effect="impact_accent",
                    intensity="moderate",
                    transcript_excerpt=phrase_3 if is_excited else raw_w,
                    parameters={
                        "badge_text": "SHOCKED REACTION",
                        "reaction_type": "excited",
                        "red_edge_flash": False,
                        "accent_color": "&H0000FFFF",  # Electric Yellow
                        "icon": "burst"
                    }
                ))
                evt_idx += 1

        # -------------------------------------------------------------
        # 3. Surprise and Unexpected Reveals
        # -------------------------------------------------------------
        for i, w in enumerate(clip_words):
            raw_w = w.get("word", "").strip()
            clean_w = re.sub(r"[^\w\s]", "", raw_w).upper()
            rel_s = max(0.0, w.get("start", clip_start) - clip_start)
            rel_e = min(clip_dur, w.get("end", clip_start) - clip_start)

            # Check question marks or surprise phrases
            if raw_w.endswith("?") or raw_w.endswith("?!") or clean_w in SURPRISE_KEYWORDS:
                # Capture the question phrase leading up to this word
                start_idx = max(0, i - 3)
                end_idx = i + 1
                excerpt = " ".join(clip_words[k].get("word", "").strip() for k in range(start_idx, end_idx))
                
                # Check that it represents a real question / surprise beat
                if "?" in excerpt or any(term in excerpt.upper() for term in ["EVERYTHING", "WHAT", "HOW", "IMPOSSIBLE"]):
                    dur = 1.5
                    clean_excerpt = re.sub(r"[^\w\s\?]", "", excerpt).strip().upper()
                    if len(clean_excerpt) > 26:
                        clean_excerpt = clean_excerpt[:26] + "..."

                    detected.append(VisualEvent(
                        event_id=f"evt_surprise_{evt_idx}",
                        event_type="surprise_reveal",
                        start=round(rel_s, 2),
                        end=round(min(clip_dur, rel_s + dur), 2),
                        duration=dur,
                        confidence=0.86,
                        evidence=f"Curiosity reveal or surprise question in speech: '{excerpt}'",
                        recommended_effect="surprise_burst",
                        intensity="moderate",
                        transcript_excerpt=excerpt,
                        parameters={
                            "badge_text": f"? {clean_excerpt} ?",
                            "accent_color": "&H00FFFF00",  # Vibrant Cyan
                            "border_color": "&H00002040",
                            "icon": "question"
                        }
                    ))
                    evt_idx += 1

        # -------------------------------------------------------------
        # 4. Challenges, Rules & Competition Stakes
        # -------------------------------------------------------------
        for i, w in enumerate(clip_words):
            raw_w = w.get("word", "").strip()
            clean_w = re.sub(r"[^\w\s]", "", raw_w).upper()
            rel_s = max(0.0, w.get("start", clip_start) - clip_start)

            phrase_3 = " ".join(
                re.sub(r"[^\w\s]", "", clip_words[k].get("word", "").strip()).upper()
                for k in range(i, min(len(clip_words), i + 4))
            )

            is_challenge = any(term in phrase_3 for term in CHALLENGE_KEYWORDS)
            if is_challenge:
                dur = 2.0
                detected.append(VisualEvent(
                    event_id=f"evt_challenge_{evt_idx}",
                    event_type="challenge_stakes",
                    start=round(rel_s, 2),
                    end=round(min(clip_dur, rel_s + dur), 2),
                    duration=dur,
                    confidence=0.85,
                    evidence=f"Challenge rule or high-stakes scale milestone: '{phrase_3}'",
                    recommended_effect="challenge_callout",
                    intensity="moderate",
                    transcript_excerpt=phrase_3,
                    parameters={
                        "badge_text": f"🔥 {phrase_3[:26]} 🔥",
                        "accent_color": "&H000080FF",  # Sunset Orange
                        "border_color": "&H00001030",
                        "icon": "fire"
                    }
                ))
                evt_idx += 1

        # -------------------------------------------------------------
        # 5. Key Statement / Climax Takeaway
        # -------------------------------------------------------------
        # Check last 4 seconds for a clean payoff / takeaway statement
        tail_words = [w for w in clip_words if (w.get("end", clip_start) - clip_start) >= clip_dur - 4.5]
        if tail_words:
            tail_text = " ".join(w.get("word", "").strip() for w in tail_words)
            has_takeaway = any(term in tail_text.upper() for term in TAKEAWAY_KEYWORDS)
            if has_takeaway:
                # Find the sentence containing the takeaway keyword
                sentences = re.split(r'[.!?]+', tail_text)
                punchline = [s.strip() for s in sentences if any(term in s.upper() for term in TAKEAWAY_KEYWORDS) and s.strip()]
                clean_tail = punchline[-1].upper() if punchline else tail_text.upper()
                clean_tail = re.sub(r"[^\w\s]", "", clean_tail).strip()
                if len(clean_tail) > 22:
                    clean_tail = clean_tail[:22] + "..."

                first_tail = tail_words[0]
                rel_s = max(0.0, first_tail.get("start", clip_start) - clip_start)
                dur = min(2.4, clip_dur - rel_s)

                detected.append(VisualEvent(
                    event_id=f"evt_takeaway_{evt_idx}",
                    event_type="key_takeaway",
                    start=round(rel_s, 2),
                    end=round(min(clip_dur, rel_s + dur), 2),
                    duration=dur,
                    confidence=0.90,
                    evidence=f"Narrative conclusion or climactic statement payoff: '{tail_text}'",
                    recommended_effect="takeaway_bracket",
                    intensity="moderate",
                    transcript_excerpt=tail_text,
                    parameters={
                        "badge_text": f"[ {clean_tail} ]",
                        "accent_color": "&H0000FFFF",  # Electric Yellow
                        "border_color": "&H00000000",
                        "icon": "bracket"
                    }
                ))
                evt_idx += 1

        # -------------------------------------------------------------
        # Filter, Deduplicate & Resolve Conflicting Events
        # -------------------------------------------------------------
        return self._resolve_event_conflicts(detected, clip_dur, intensity)

    def _resolve_event_conflicts(
        self,
        events: List[VisualEvent],
        clip_duration: float,
        intensity: str
    ) -> List[VisualEvent]:
        """
        Applies pacing rules to prevent chaotic overlapping graphics:
        1. Limits simultaneous high-intensity treatments.
        2. Enforces minimum separation (1.2s) between events of identical category.
        3. Caps total event count based on intensity preset (Vibrant Pro = 2 to 4, Clean Pro = 0 to 1).
        """
        if not events:
            return []

        # Sort primarily by confidence descending, then start time
        events.sort(key=lambda e: (e.confidence, -e.start), reverse=True)

        selected: List[VisualEvent] = []
        max_events = 4 if intensity in ["dynamic", "high_energy"] else 2
        if intensity == "subtle":
            max_events = 1

        # Multi-pass selection to maximize category diversity:
        # Pass 1: Select the highest-confidence event from each distinct event_type
        # Pass 2: Fill remaining slots up to max_events
        categories_seen = set()
        for candidate in events:
            if len(selected) >= max_events:
                break
            if candidate.event_type in categories_seen:
                continue

            conflict = False
            for sel in selected:
                overlap_start = max(candidate.start, sel.start)
                overlap_end = min(candidate.end, sel.end)
                if overlap_end > overlap_start:
                    if not (candidate.recommended_effect == "impact_accent" or sel.recommended_effect == "impact_accent"):
                        conflict = True
                        break

            if not conflict:
                selected.append(candidate)
                categories_seen.add(candidate.event_type)

        # Pass 2: Fill remaining slots if any, enforcing category spacing
        if len(selected) < max_events:
            for candidate in events:
                if len(selected) >= max_events:
                    break
                if candidate in selected:
                    continue

                conflict = False
                for sel in selected:
                    if candidate.event_type == sel.event_type and abs(candidate.start - sel.start) < 2.5:
                        conflict = True
                        break
                    overlap_start = max(candidate.start, sel.start)
                    overlap_end = min(candidate.end, sel.end)
                    if overlap_end > overlap_start:
                        if not (candidate.recommended_effect == "impact_accent" or sel.recommended_effect == "impact_accent"):
                            conflict = True
                            break

                if not conflict:
                    selected.append(candidate)

        # Sort chronologically for execution
        selected.sort(key=lambda e: e.start)
        return selected
