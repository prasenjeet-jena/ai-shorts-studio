from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.events import VisualEvent


def format_ass_time(seconds: float) -> str:
    """Formats float seconds into ASS timestamp format: H:MM:SS.cc"""
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        secs += 1
        centis -= 100
    return f"{hrs}:{mins:02d}:{secs:02d}.{centis:02d}"


@dataclass
class EffectDefinition:
    """
    Specification for a visual effect template.
    """
    name: str
    supported_events: List[str]
    style_name: str
    font_name: str
    font_size: int
    primary_color: str      # ASS hex: &HAABBGGRR
    outline_color: str
    shadow_color: str
    outline_width: float
    shadow_depth: float
    alignment: int          # 8 = top center, 5 = middle center
    margin_v: int           # Vertical margin from top or bottom (px)
    default_duration: float
    enabled_presets: List[str]


# Modular Registry of Visual Effects
EFFECT_REGISTRY: Dict[str, EffectDefinition] = {
    "money_reward_badge": EffectDefinition(
        name="money_reward_badge",
        supported_events=["money_reward"],
        style_name="EvtMoney",
        font_name="Impact",
        font_size=52,
        primary_color="&H0000FFFF",   # Electric Gold / Yellow
        outline_color="&H00004000",   # Emerald Dark Border
        shadow_color="&HA0000000",    # Semi-transparent Drop Shadow
        outline_width=4.5,
        shadow_depth=3.0,
        alignment=8,                  # Top-Center safe header zone
        margin_v=240,                 # Clears top UI notch / status bar
        default_duration=2.2,
        enabled_presets=["clean_pro", "vibrant_pro", "high_energy", "story_remix"]
    ),
    "surprise_burst": EffectDefinition(
        name="surprise_burst",
        supported_events=["surprise_reveal"],
        style_name="EvtSurprise",
        font_name="Impact",
        font_size=50,
        primary_color="&H00FFFF00",   # Vibrant Cyan
        outline_color="&H00002040",   # Deep Navy Border
        shadow_color="&HA0000000",
        outline_width=4.2,
        shadow_depth=3.0,
        alignment=8,
        margin_v=240,
        default_duration=1.6,
        enabled_presets=["vibrant_pro", "high_energy", "story_remix"]
    ),
    "impact_accent": EffectDefinition(
        name="impact_accent",
        supported_events=["intense_reaction"],
        style_name="EvtImpact",
        font_name="Impact",
        font_size=58,
        primary_color="&H000000FF",   # Crimson Red
        outline_color="&H00000000",   # Black Border
        shadow_color="&H80000000",
        outline_width=4.8,
        shadow_depth=3.0,
        alignment=8,
        margin_v=300,
        default_duration=1.1,
        enabled_presets=["vibrant_pro", "high_energy", "story_remix"]
    ),
    "challenge_callout": EffectDefinition(
        name="challenge_callout",
        supported_events=["challenge_stakes"],
        style_name="EvtChallenge",
        font_name="Impact",
        font_size=46,
        primary_color="&H000080FF",   # Sunset Warm Orange
        outline_color="&H00001030",   # Dark Maroon Outline
        shadow_color="&HA0000000",
        outline_width=4.0,
        shadow_depth=2.5,
        alignment=8,
        margin_v=240,
        default_duration=2.0,
        enabled_presets=["vibrant_pro", "high_energy", "story_remix"]
    ),
    "takeaway_bracket": EffectDefinition(
        name="takeaway_bracket",
        supported_events=["key_takeaway"],
        style_name="EvtTakeaway",
        font_name="Impact",
        font_size=48,
        primary_color="&H0000FFFF",   # Electric Yellow
        outline_color="&H00000000",   # Pure Black
        shadow_color="&HA0000000",
        outline_width=4.2,
        shadow_depth=2.5,
        alignment=8,
        margin_v=250,
        default_duration=2.0,
        enabled_presets=["clean_pro", "vibrant_pro", "high_energy", "story_remix"]
    ),
    "celebration_burst": EffectDefinition(
        name="celebration_burst",
        supported_events=["celebration_win"],
        style_name="EvtCelebration",
        font_name="Impact",
        font_size=54,
        primary_color="&H0000FF66",   # Neon Emerald
        outline_color="&H00003010",
        shadow_color="&H90000000",
        outline_width=4.5,
        shadow_depth=3.0,
        alignment=8,
        margin_v=240,
        default_duration=2.2,
        enabled_presets=["clean_pro", "vibrant_pro", "high_energy", "story_remix"]
    ),
    "remix_bubble": EffectDefinition(
        name="remix_bubble",
        supported_events=["editorial_thought", "freeze_commentary"],
        style_name="RemixBubble",
        font_name="Impact",
        font_size=48,
        primary_color="&H00FFFFFF",   # Crisp White fill
        outline_color="&H00000000",   # Solid Black comic ink border
        shadow_color="&HA0000000",
        outline_width=5.0,
        shadow_depth=3.5,
        alignment=8,                  # Upper safe header zone
        margin_v=240,
        default_duration=1.2,
        enabled_presets=["story_remix"]
    ),
    "remix_aside": EffectDefinition(
        name="remix_aside",
        supported_events=["editorial_aside"],
        style_name="RemixAside",
        font_name="Impact",
        font_size=48,
        primary_color="&H0000FFFF",   # Punchy Comic Yellow
        outline_color="&H00000000",   # Bold Black Outline
        shadow_color="&HA0000000",
        outline_width=4.8,
        shadow_depth=3.0,
        alignment=8,
        margin_v=240,
        default_duration=1.5,
        enabled_presets=["story_remix"]
    ),
    "remix_cartoon": EffectDefinition(
        name="remix_cartoon",
        supported_events=["cartoon_accent"],
        style_name="RemixCartoon",
        font_name="Arial",
        font_size=62,
        primary_color="&H00FFFFFF",
        outline_color="&H00000000",
        shadow_color="&HA0000000",
        outline_width=4.0,
        shadow_depth=2.5,
        alignment=8,
        margin_v=210,
        default_duration=1.4,
        enabled_presets=["story_remix"]
    )
}


class VisualEffectEngine:
    """
    Compositing engine for context-aware visual graphics:
    - Generates ASS graphic overlay dialogues with animated pop-in and fade-out easing.
    - Generates FFmpeg video filter expressions (e.g. momentary red impact edge flash).
    - Respects mobile safe margins and intensity presets (Clean Pro, Vibrant Pro, High Energy).
    """

    def __init__(self):
        pass

    def get_ass_styles(self, preset: str = "vibrant_pro") -> List[str]:
        """
        Returns ASS V4+ Style definition lines for all registered effects compatible with the preset.
        """
        lines = []
        preset_clean = preset.lower()

        for eff in EFFECT_REGISTRY.values():
            if preset_clean not in eff.enabled_presets:
                continue

            # Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour,
            # Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline,
            # Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
            line = (
                f"Style: {eff.style_name},{eff.font_name},{eff.font_size},"
                f"{eff.primary_color},&H00FFFFFF,{eff.outline_color},{eff.shadow_color},"
                f"-1,0,0,0,100,100,0,0,1,{eff.outline_width},{eff.shadow_depth},"
                f"{eff.alignment},24,24,{eff.margin_v},1"
            )
            lines.append(line)

        return lines

    def compile_ass_events(
        self,
        events: List[VisualEvent],
        clip_start: float = 0.0,
        preset: str = "vibrant_pro"
    ) -> List[str]:
        """
        Compiles detected VisualEvents into animated ASS Dialogue lines on Layer 2 (overlays).
        """
        dialogue_lines: List[str] = []
        preset_clean = preset.lower()

        for evt in events:
            eff_def = EFFECT_REGISTRY.get(evt.recommended_effect)
            if not eff_def or preset_clean not in eff_def.enabled_presets:
                continue

            start_t = evt.start
            end_t = evt.end
            dur_ms = int(evt.duration * 1000)

            ass_start = format_ass_time(start_t)
            ass_end = format_ass_time(end_t)

            # Easing: pop-in (100% -> 118% -> 100%) in first 240ms, fade-out in last 220ms
            pop_dur = 140 if preset_clean == "high_energy" else 180
            fade_start_ms = max(0, dur_ms - 220)

            anim_tags = (
                f"\\an{eff_def.alignment}"
                f"\\t(0,{pop_dur},\\fscx118\\fscy118)"
                f"\\t({pop_dur},{pop_dur+100},\\fscx100\\fscy100)"
                f"\\t({fade_start_ms},{dur_ms},\\alpha&HFF&)"
            )

            # Badge display text
            badge_text = evt.parameters.get("badge_text", evt.transcript_excerpt.upper())
            # Clean and wrap with stylish brackets if not already present
            if not (badge_text.startswith("[") or badge_text.startswith("【") or badge_text.startswith("?")):
                badge_text = f"[ {badge_text} ]"

            dialogue = f"Dialogue: 2,{ass_start},{ass_end},{eff_def.style_name},,0,0,0,,{{{anim_tags}}}{badge_text}"
            dialogue_lines.append(dialogue)

        return dialogue_lines

    def compile_video_filters(
        self,
        events: List[VisualEvent],
        preset: str = "vibrant_pro"
    ) -> List[str]:
        """
        Generates FFmpeg video filter elements for high-impact effects:
        - Momentary red impact accent edge frame during confrontation/panic beats.
        """
        filters: List[str] = []
        preset_clean = preset.lower()

        if preset_clean in ["clean_pro", "clean"]:
            # Clean Pro disables video filter flashes
            return []

        for evt in events:
            # Check for red edge flash in parameters or intense reaction confrontation
            if evt.parameters.get("red_edge_flash") or (evt.event_type == "intense_reaction" and evt.intensity == "high"):
                s = evt.start
                # Keep flash brief (0.5s - 0.7s) to avoid strobing
                e = min(evt.end, s + 0.65)
                # 22px red border frame
                f_expr = (
                    f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.32:t=22:enable=\x27between(t\\,{s:.2f}\\,{e:.2f})\x27"
                )
                filters.append(f_expr)

        return filters

    def compile_storyboard_ass_events(
        self,
        storyboard: Any,
        clip_duration: float,
        minimal_text: bool = False
    ) -> List[str]:
        """
        Compiles a StoryRemixStoryboard into Layer 2 animated ASS Dialogue lines.
        When minimal_text=True, only the freeze-frame thought bubble is emitted,
        allowing full-frame video animation (cel-shading, speed lines, bursts) to lead.
        """
        dialogue_lines: List[str] = []

        sb_dict = storyboard.to_dict() if hasattr(storyboard, "to_dict") else storyboard
        if not sb_dict:
            return []

        # 1. Opening Hook Banner (omitted when minimal_text=True)
        hook = sb_dict.get("opening_hook", {})
        if not minimal_text and hook and hook.get("title"):
            s_t = hook.get("start_time", 0.3)
            e_t = min(hook.get("end_time", 2.6), clip_duration)
            ass_s = format_ass_time(s_t)
            ass_e = format_ass_time(e_t)
            dur_ms = int((e_t - s_t) * 1000)
            fade_ms = max(0, dur_ms - 250)
            anim = f"\\an8\\t(0,180,\\fscx115\\fscy115)\\t(180,260,\\fscx100\\fscy100)\\t({fade_ms},{dur_ms},\\alpha&HFF&)"
            dialogue_lines.append(f"Dialogue: 2,{ass_s},{ass_e},RemixAside,,0,0,0,,{{{anim}}}{hook['title']}")

        # 2. Freeze-Frame Editorial Commentary
        freeze = sb_dict.get("freeze_beat")
        if freeze and freeze.get("editorial_commentary"):
            f_start = freeze.get("start_time", 0.0)
            f_dur = freeze.get("duration", 1.0)
            f_end = f_start + f_dur
            ass_s = format_ass_time(f_start)
            ass_e = format_ass_time(f_end)
            dur_ms = int(f_dur * 1000)
            commentary = freeze["editorial_commentary"].strip()

            style = "RemixBubble" if freeze.get("presentation_style") == "thought_bubble" else "RemixAside"
            anim = f"\\an8\\t(0,150,\\fscx112\\fscy112)\\t(150,240,\\fscx100\\fscy100)"
            if style == "RemixBubble" and not commentary.startswith("("):
                commentary = f"(THOUGHT: {commentary})"
            elif style == "RemixAside" and not commentary.startswith("["):
                commentary = f"[ EDITORIAL: {commentary} ]"

            dialogue_lines.append(f"Dialogue: 2,{ass_s},{ass_e},{style},,0,0,0,,{{{anim}}}{commentary}")

        # 3. Cartoon Accents (omitted when minimal_text=True to let video animation lead)
        if not minimal_text:
            for accent in sb_dict.get("cartoon_accents", []):
                s_t = accent.get("start_time", 0.0)
                e_t = min(accent.get("end_time", s_t + 1.2), clip_duration)
                if s_t >= clip_duration:
                    continue
                ass_s = format_ass_time(s_t)
                ass_e = format_ass_time(e_t)
                sym = accent.get("symbol", "!!")
                lbl = accent.get("label", "")
                dur_ms = int((e_t - s_t) * 1000)
                fade_ms = max(0, dur_ms - 200)

                acc_type = accent.get("accent_type", "")
                if acc_type == "sweat_drop":
                    text = f"[ (!) {lbl} (!) ]"
                    anim = f"\\an8\\pos(540,240)\\t(0,180,\\fscx120\\fscy120)\\t(180,260,\\fscx100\\fscy100)\\t({fade_ms},{dur_ms},\\alpha&HFF&)"
                    dialogue_lines.append(f"Dialogue: 2,{ass_s},{ass_e},RemixAside,,0,0,0,,{{{anim}}}{text}")
                elif acc_type == "reaction_burst":
                    text = f"!! {lbl} !!"
                    anim = f"\\an8\\pos(540,240)\\t(0,140,\\fscx135\\fscy135)\\t(140,240,\\fscx100\\fscy100)\\t({fade_ms},{dur_ms},\\alpha&HFF&)"
                    dialogue_lines.append(f"Dialogue: 2,{ass_s},{ass_e},RemixAside,,0,0,0,,{{{anim}}}{text}")
                else:
                    text = f"{sym} {lbl} {sym}".strip()
                    anim = f"\\an8\\t(0,180,\\fscx115\\fscy115)\\t(180,260,\\fscx100\\fscy100)\\t({fade_ms},{dur_ms},\\alpha&HFF&)"
                    dialogue_lines.append(f"Dialogue: 2,{ass_s},{ass_e},RemixAside,,0,0,0,,{{{anim}}}{text}")

            # 4. Climax Payoff
            climax = sb_dict.get("climax_payoff", {})
            if climax and climax.get("text"):
                s_t = climax.get("start_time", max(0.0, clip_duration - 2.5))
                e_t = min(climax.get("end_time", clip_duration), clip_duration)
                ass_s = format_ass_time(s_t)
                ass_e = format_ass_time(e_t)
                dur_ms = int((e_t - s_t) * 1000)
                fade_ms = max(0, dur_ms - 200)
                anim = f"\\an8\\t(0,180,\\fscx115\\fscy115)\\t(180,260,\\fscx100\\fscy100)\\t({fade_ms},{dur_ms},\\alpha&HFF&)"
                dialogue_lines.append(f"Dialogue: 2,{ass_s},{ass_e},RemixAside,,0,0,0,,{{{anim}}}{climax['text']}")

        return dialogue_lines

