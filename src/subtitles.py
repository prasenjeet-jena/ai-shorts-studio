import re
from pathlib import Path
from typing import List, Dict, Any, Optional

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

COLOR_WHITE = "&H00FFFFFF"
COLOR_YELLOW = "&H0000FFFF"    # Electric Yellow
COLOR_CYAN = "&H00FFFF00"      # Vibrant Cyan
COLOR_ORANGE = "&H000080FF"    # Warm Sunset Orange
COLOR_GREEN = "&H0000FF66"     # Neon Lime Green
COLOR_MAGENTA = "&H00FF00FF"   # Electric Magenta
COLOR_BLACK = "&H00000000"     # Pure Black Outline

# High-impact creator keywords that warrant visual emphasis
KEYWORD_LIST = {
    "INTERNET", "HONEST", "CARE", "SEEN", "SHARED", "WRONG", "FACEBOOK",
    "AUDIENCE", "TALK", "THINK", "PEOPLE", "CLICK", "ATTENTION", "SPANS",
    "INFLATION", "EXPENSIVE", "RANDOM", "TRUTH", "SECRET", "MONEY", "STOP",
    "NEVER", "ALWAYS", "BECAUSE", "WORST", "BEST", "CRAZY", "INSANE", "REAL"
}

def is_emphasis_keyword(word_text: str, duration: float = 0.0) -> bool:
    """Detects if a word is an emphatic keyword based on lexicon or duration."""
    clean = re.sub(r"[^\w]", "", word_text).upper()
    if clean in KEYWORD_LIST:
        return True
    # Emphatic delivery: longer duration on concise words
    if len(clean) >= 3 and duration >= 0.44:
        return True
    return False

class SubtitleGenerator:
    """
    Advanced Phase 3 ASS Subtitle Generator with:
    - Animated phrase pop-in with smooth easing.
    - Active-word synchronized highlighting with contrasting colors.
    - Keyword emphasis with accent color pops and underline bars.
    - Three reusable production presets: Clean Pro, Vibrant Pro, High Energy.
    - Mobile safe margin positioning (clears platform UI buttons and lower-thirds).
    """

    def __init__(self):
        pass

    def build_natural_phrases(
        self,
        words: List[Dict[str, Any]],
        max_words_per_phrase: int = 5,
        min_words_per_phrase: int = 2
    ) -> List[List[Dict[str, Any]]]:
        if not words:
            return []

        phrases = []
        curr = []

        for i, w in enumerate(words):
            curr.append(w)
            w_text = w.get("word", "").strip()

            has_punct = any(w_text.endswith(p) for p in [".", "!", "?", ",", ";", ":", "—"])
            
            has_pause = False
            if i < len(words) - 1:
                gap = words[i + 1].get("start", 0) - w.get("end", 0)
                if gap > 0.28:
                    has_pause = True

            is_full = len(curr) >= max_words_per_phrase
            remaining = len(words) - (i + 1)

            if (has_punct or has_pause or is_full) and len(curr) >= min_words_per_phrase:
                if remaining != 1:
                    phrases.append(curr)
                    curr = []

        if curr:
            if phrases and len(curr) < min_words_per_phrase:
                phrases[-1].extend(curr)
            else:
                phrases.append(curr)

        return phrases

    def generate_ass(
        self,
        words_data: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float,
        output_file: Path,
        preset: str = "vibrant_pro",
        style: Optional[str] = None,
        font_name: str = "Arial",
        font_size: Optional[int] = None,
        margin_v: Optional[int] = None,
        accent_color: Optional[str] = None,
        primary_color: str = COLOR_WHITE,
        outline_color: str = COLOR_BLACK,
        outline_width: Optional[int] = None,
        shadow_offset: Optional[int] = None,
        word_highlighting: bool = True,
        visual_events: Optional[List[Any]] = None,
        storyboard: Optional[Any] = None,
        comedy_timeline: Optional[Any] = None,
        **kwargs
    ) -> Path:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        clip_words = [
            w for w in words_data
            if w.get("end", 0) >= clip_start and w.get("start", 0) <= clip_end
        ]

        # Normalize preset / style aliases
        selected_mode = (style or preset).lower()
        if selected_mode in ["clean", "clean_pro"]:
            preset_name = "clean_pro"
            base_font_size = font_size or 58
            base_margin_v = margin_v or 420
            base_outline = outline_width or 3
            base_shadow = shadow_offset or 1
            active_accent = accent_color or COLOR_YELLOW
            keyword_accent = COLOR_YELLOW
            use_pop_in = False
            pop_scale = 100
            use_underline = False
            use_word_highlight = True
        elif selected_mode in ["high_energy", "energy"]:
            preset_name = "high_energy"
            base_font_size = font_size or 72
            base_margin_v = margin_v or 440
            base_outline = outline_width or 6
            base_shadow = shadow_offset or 3
            active_accent = accent_color or COLOR_YELLOW
            keyword_accent = COLOR_CYAN
            use_pop_in = True
            pop_scale = 118
            use_underline = True
            use_word_highlight = True
        elif selected_mode in ["full_cartoon", "cartoon", "anime"]:
            preset_name = "full_cartoon"
            font_name = "Impact"
            base_font_size = font_size or 72
            base_margin_v = margin_v or 240
            base_outline = outline_width or 7
            base_shadow = shadow_offset or 3
            active_accent = accent_color or COLOR_YELLOW
            keyword_accent = COLOR_YELLOW
            use_pop_in = True
            pop_scale = 116
            use_underline = False
            use_word_highlight = True
        elif selected_mode in ["full_comedy_remix", "comedy_remix"]:
            preset_name = "full_comedy_remix"
            font_name = "Impact"
            base_font_size = font_size or 72
            base_margin_v = margin_v or 240
            base_outline = outline_width or 7
            base_shadow = shadow_offset or 3
            active_accent = accent_color or COLOR_YELLOW
            keyword_accent = COLOR_YELLOW
            use_pop_in = True
            pop_scale = 116
            use_underline = False
            use_word_highlight = True
        elif selected_mode in ["story_remix", "remix"]:
            preset_name = "story_remix"
            font_name = "Impact"
            base_font_size = font_size or 70
            base_margin_v = margin_v or 430
            base_outline = outline_width or 6
            base_shadow = shadow_offset or 3
            active_accent = accent_color or COLOR_YELLOW
            keyword_accent = COLOR_CYAN
            use_pop_in = True
            pop_scale = 115
            use_underline = True
            use_word_highlight = True
        elif selected_mode == "cinematic":
            preset_name = "cinematic"
            font_name = "Georgia"
            base_font_size = font_size or 52
            base_margin_v = margin_v or 360
            base_outline = outline_width or 2
            base_shadow = shadow_offset or 0
            primary_color = "&H00EEEEEE"
            active_accent = COLOR_WHITE
            keyword_accent = COLOR_WHITE
            use_pop_in = False
            pop_scale = 100
            use_underline = False
            use_word_highlight = False
        else:
            # Default: Vibrant Pro
            preset_name = "vibrant_pro"
            base_font_size = font_size or 68
            base_margin_v = margin_v or 440
            base_outline = outline_width or 5
            base_shadow = shadow_offset or 2
            active_accent = accent_color or COLOR_YELLOW
            keyword_accent = COLOR_CYAN
            use_pop_in = True
            pop_scale = 112
            use_underline = True
            use_word_highlight = word_highlighting

        ass_header = f"""[Script Info]
Title: AI Shorts Vibrant Pro Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{base_font_size},{primary_color},{active_accent},{outline_color},&H80000000,-1,0,0,0,100,100,0,0,1,{base_outline},{base_shadow},2,60,60,{base_margin_v},1
"""
        from src.effects import VisualEffectEngine
        eff_engine = VisualEffectEngine()
        eff_styles = eff_engine.get_ass_styles(preset=preset_name)
        if eff_styles:
            ass_header += "\n".join(eff_styles) + "\n"

        ass_header += """
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events = []

        if not clip_words:
            events.append("Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(ass_header + "\n".join(events) + "\n")
            return output_file

        phrases = self.build_natural_phrases(clip_words)

        for p_idx, phrase in enumerate(phrases):
            p_start_rel = max(0.0, phrase[0]["start"] - clip_start)
            if p_idx < len(phrases) - 1:
                next_p_start = max(0.0, phrases[p_idx + 1][0]["start"] - clip_start)
                p_end_rel = min(clip_end - clip_start, phrase[-1]["end"] - clip_start + 0.15, next_p_start)
            else:
                p_end_rel = min(clip_end - clip_start, phrase[-1]["end"] - clip_start + 0.25)

            if p_end_rel <= p_start_rel:
                continue

            # Pop-in transform header tag applied to phrase entrance
            pop_tag = ""
            if use_pop_in:
                pop_tag = f"{{\\fscx{pop_scale}\\fscy{pop_scale}\\t(0,90,\\fscx100\\fscy100)}}"

            if use_word_highlight:
                for w_idx, active_word in enumerate(phrase):
                    w_start_rel = max(p_start_rel, active_word["start"] - clip_start)
                    if w_idx < len(phrase) - 1:
                        next_w_start = max(w_start_rel, phrase[w_idx + 1]["start"] - clip_start)
                        w_end_rel = min(p_end_rel, next_w_start)
                    else:
                        w_end_rel = p_end_rel

                    if w_end_rel <= w_start_rel:
                        w_end_rel = min(p_end_rel, w_start_rel + 0.05)

                    line_words = []
                    for idx, w in enumerate(phrase):
                        w_raw = w["word"]
                        w_text = w_raw.upper()
                        w_dur = w.get("end", 0) - w.get("start", 0)
                        is_kw = is_emphasis_keyword(w_raw, w_dur)

                        if idx == w_idx:
                            # Active word
                            if is_kw and preset_name in ["vibrant_pro", "high_energy"]:
                                color = keyword_accent
                                u_tag = "\\u1" if use_underline else ""
                                line_words.append(f"{{\\c{color}&{u_tag}}}{w_text}{{\\u0\\c{primary_color}&}}")
                            else:
                                line_words.append(f"{{\\c{active_accent}&}}{w_text}{{\\c{primary_color}&}}")
                        else:
                            # Inactive word
                            line_words.append(w_text)

                    phrase_line = " ".join(line_words)
                    # Only the first word event in the phrase gets the entrance pop-in tag
                    if w_idx == 0 and pop_tag:
                        phrase_line = f"{pop_tag}{phrase_line}"

                    t_start_str = format_ass_time(w_start_rel)
                    t_end_str = format_ass_time(w_end_rel)
                    events.append(f"Dialogue: 0,{t_start_str},{t_end_str},Default,,0,0,0,,{phrase_line}")
            else:
                # Phrase-level display (Clean / Cinematic)
                clean_words = []
                for w in phrase:
                    w_text = w["word"]
                    if preset_name != "cinematic":
                        w_text = w_text.upper()
                    clean_words.append(w_text)
                clean_text = " ".join(clean_words)
                if pop_tag:
                    clean_text = f"{pop_tag}{clean_text}"
                t_start_str = format_ass_time(p_start_rel)
                t_end_str = format_ass_time(p_end_rel)
                events.append(f"Dialogue: 0,{t_start_str},{t_end_str},Default,,0,0,0,,{clean_text}")

        # Append animated visual effect overlays (Layer 2 & Layer 3)
        if visual_events:
            eff_dialogues = eff_engine.compile_ass_events(
                events=visual_events,
                clip_start=clip_start,
                preset=preset_name
            )
            events.extend(eff_dialogues)

        if storyboard:
            sb_dialogues = eff_engine.compile_storyboard_ass_events(
                storyboard=storyboard,
                clip_duration=clip_end - clip_start,
                minimal_text=(preset_name in ["story_remix", "remix"])
            )
            events.extend(sb_dialogues)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(ass_header + "\n".join(events) + "\n")

        return output_file
