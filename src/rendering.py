import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from src.config import config
from src.subtitles import SubtitleGenerator
from src.framing import SmartFramingEngine
from src.scene import SceneDetector
from src.pacing import PacingOptimizer
from src.events import VisualEventDetector, VisualEvent
from src.effects import VisualEffectEngine
from src.storyboard import StoryRemixEngine, StoryRemixStoryboard
from src.sound import SoundEngine
from src.anim import AnimationEngine
from src.comedy_timeline import FullComedyRemixEngine, FullComedyTimeline
from src.reactions import ReactionLibrary
from src.emoji_anim import AnimatedEmojiEngine, EmojiReaction
from src.cartoon_engine import AnimeGANv2CartoonEngine


class VideoRenderer:
    def __init__(self):
        self.sub_gen = SubtitleGenerator()
        self.framing_engine = SmartFramingEngine()
        self.scene_detector = SceneDetector()
        self.pacing_optimizer = PacingOptimizer()
        self.event_detector = VisualEventDetector()
        self.effect_engine = VisualEffectEngine()
        self.story_engine = StoryRemixEngine()
        self.sound_engine = SoundEngine()
        self.anim_engine = AnimationEngine()
        self.comedy_engine = FullComedyRemixEngine()
        self.reaction_lib = ReactionLibrary()
        self.emoji_engine = AnimatedEmojiEngine()
        self.cartoon_engine = AnimeGANv2CartoonEngine()

    def probe_video(self, video_path: Path) -> Dict[str, Any]:
        cmd = [
            config.FFPROBE_PATH,
            "-v", "error",
            "-show_entries", "stream=width,height,duration,codec_name,codec_type",
            "-of", "json",
            str(video_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        data = json.loads(res.stdout.decode('utf-8'))
        streams = data.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        return {
            "width": int(video_stream.get("width", 1920)),
            "height": int(video_stream.get("height", 1080)),
            "codec": video_stream.get("codec_name", "h264"),
            "video_codec": video_stream.get("codec_name", "h264"),
            "has_audio": audio_stream is not None,
            "audio_codec": audio_stream.get("codec_name") if audio_stream else None
        }

    def detect_zoom_intervals(
        self,
        words_data: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float,
        preset: str = "vibrant_pro",
        intensity: str = "selective"
    ) -> List[Tuple[float, float]]:
        """
        Identifies key emphasis moments for subtle punch-in zooms (1.08x).
        Uses word scoring, clustering around thesis and punchline moments,
        and enforces healthy separation between zooms to maintain comfortable viewing rhythm.
        """
        intensity_clean = intensity.lower()
        if intensity_clean in ["off", "none", "0"]:
            return []

        preset_clean = preset.lower()
        if preset_clean in ["clean_pro", "clean", "cinematic"] and intensity_clean != "dynamic":
            return []

        clip_words = [w for w in words_data if w.get("end", 0) >= clip_start and w.get("start", 0) <= clip_end]
        if not clip_words:
            return []

        from src.subtitles import KEYWORD_LIST
        clip_dur = clip_end - clip_start

        scored = []
        for w in clip_words:
            clean = "".join(c for c in w.get("word", "").upper() if c.isalnum())
            dur = w.get("end", 0) - w.get("start", 0)
            rel_s = round(w.get("start", 0) - clip_start, 2)
            rel_e = round(w.get("end", 0) - clip_start, 2)
            score = 0
            if clean in {"INTERNET", "WRONG", "FACEBOOK", "SCROLLING", "SECRET", "MILLION", "NEVER", "HONEST", "MONEY", "VIRAL", "TRUTH"}:
                score += 3
            elif clean in KEYWORD_LIST:
                score += 1.5
            if dur > 0.45:
                score += 1
            if score >= 2:
                scored.append((rel_s, rel_e, clean, score))

        if not scored:
            return []

        # Cluster words within 2.5s of each other
        clusters = []
        current_cluster = [scored[0]]
        for item in scored[1:]:
            if item[0] - current_cluster[-1][1] <= 2.5:
                current_cluster.append(item)
            else:
                clusters.append(current_cluster)
                current_cluster = [item]
        clusters.append(current_cluster)

        # Convert clusters to intervals with tasteful padding
        intervals = []
        for cl in clusters:
            s = max(0.5, cl[0][0] - 0.3)
            e = min(clip_dur, cl[-1][1] + 1.2)
            dur = e - s
            if dur < 2.0:
                e = min(clip_dur, s + 2.2)
            elif dur > 4.8:
                e = s + 4.8
            total_score = sum(x[3] for x in cl)
            intervals.append({"start": round(s, 2), "end": round(e, 2), "score": total_score})

        intervals.sort(key=lambda x: x["score"], reverse=True)
        max_zooms = 3 if (preset_clean in ["high_energy", "energy"] or intensity_clean == "dynamic") else 2
        min_gap = 1.8 if (preset_clean in ["high_energy", "energy"] or intensity_clean == "dynamic") else 2.5

        selected = []
        for it in intervals:
            too_close = False
            for sel in selected:
                if not (it["end"] + min_gap <= sel["start"] or it["start"] >= sel["end"] + min_gap):
                    too_close = True
                    break
            if not too_close:
                selected.append(it)
            if len(selected) >= max_zooms:
                break

        selected.sort(key=lambda x: x["start"])
        return [(x["start"], x["end"]) for x in selected]

    def render_clip(
        self,
        source_video: Path,
        words_data: List[Dict[str, Any]],
        clip_spec: Dict[str, Any],
        output_dir: Optional[Path] = None,
        preset: str = "vibrant_pro",
        style: Optional[str] = None,
        framing: str = "auto",
        font_name: str = "Arial",
        font_size: Optional[int] = None,
        accent_color: Optional[str] = None,
        margin_v: Optional[int] = None,
        normalize_audio: bool = True,
        editing_intensity: str = "balanced",
        silence_trim: str = "conservative",
        zoom_intensity: str = "selective",
        scene_detect: bool = True,
        reuse_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Phase 4 Intelligent Editing Pipeline with Phase 6 Output Protection:
        1. Checks for existing verified output to allow instantaneous resumption.
        2. Intelligently trims sluggish dead air while preserving timing and setups.
        3. Remaps word timestamps onto edited timeline.
        4. Applies 9:16 vertical smart crop and dynamic punch zooms.
        5. Burns animated ASS subtitles.
        6. Normalizes broadcast loudness to -16.5 LUFS.
        7. Writes to temporary file and validates via probe before atomic swap.
        """
        out_dir = output_dir or config.OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(words_data, dict) and "data" in words_data:
            words_data = words_data["data"]

        clip_id = clip_spec.get("id", "clip_1")
        start = float(clip_spec["start"])
        end = float(clip_spec["end"])
        duration = round(end - start, 2)
        clip_preset = clip_spec.get("preset", preset or "vibrant_pro")
        clip_style = clip_spec.get("style", style)
        clip_framing = clip_spec.get("framing", framing)

        # Map editing_intensity overrides if not explicitly overridden
        edit_intensity = clip_spec.get("editing_intensity", editing_intensity)
        active_silence_trim = clip_spec.get("silence_trim", silence_trim)
        active_zoom_intensity = clip_spec.get("zoom_intensity", zoom_intensity)

        if edit_intensity == "subtle":
            active_silence_trim = "conservative"
            active_zoom_intensity = "selective"
        elif edit_intensity == "dynamic":
            active_silence_trim = "moderate"
            active_zoom_intensity = "dynamic"

        out_filename = f"{clip_id}.mp4"
        out_file = out_dir / out_filename

        # Phase 6 Output Reuse: If output already exists and passes validation, reuse without re-rendering
        if reuse_existing and out_file.exists():
            try:
                probe = self.probe_video(out_file)
                if (probe["width"] == 1080 and probe["height"] == 1920) and out_file.stat().st_size > 5000:
                    filesize_mb = round(out_file.stat().st_size / (1024 * 1024), 2)
                    return {
                        "id": clip_id,
                        "filename": out_filename,
                        "filepath": str(out_file.resolve()),
                        "duration": duration,
                        "original_duration": duration,
                        "dead_air_trimmed": 0.0,
                        "trim_count": 0,
                        "filesize_mb": filesize_mb,
                        "resolution": f"{probe['width']}x{probe['height']}",
                        "preset": clip_preset,
                        "editing_intensity": edit_intensity,
                        "silence_trim": active_silence_trim,
                        "zoom_intensity": active_zoom_intensity,
                        "framing_layout": "reused",
                        "framing_reason": "Reused verified existing output",
                        "zoom_intervals": [],
                        "scene_cuts": [],
                        "hook": clip_spec.get("hook", ""),
                        "hook_type": clip_spec.get("hook_type", "declarative"),
                        "reason": clip_spec.get("reason", ""),
                        "reused": True
                    }
            except Exception:
                pass  # Fall through to re-render if corrupt

        # 1. Scene detection
        scene_cuts: List[float] = []
        if scene_detect:
            scene_cuts = self.scene_detector.detect_scene_cuts(source_video, start, duration)

        # 2. Pacing and dead-air optimization
        pacing_res = self.pacing_optimizer.optimize_pacing(
            words_data=words_data,
            clip_start=start,
            clip_end=end,
            mode=active_silence_trim,
            scene_cuts=scene_cuts
        )

        is_trimmed = pacing_res["is_trimmed"]
        keep_segs = pacing_res["keep_segments"]
        active_words = pacing_res["remapped_words"]
        render_dur = pacing_res["new_duration"]

        # Cartoon filter logic removed: preserve authentic footage with motion, designs & emojis
        if clip_preset in ["full_cartoon", "cartoon", "anime", "tom_and_jerry", "classic_toon"]:
            clip_preset = "full_comedy_remix"
        is_comedy_remix = clip_preset in ["full_comedy_remix", "comedy_remix"]
        is_story_remix = clip_preset in ["story_remix", "remix"]
        is_remix = is_story_remix or is_comedy_remix
        storyboard = None
        comedy_timeline = None
        sfx_path = None
        freeze_beat = None
        reaction_path = None
        render_items: List[Tuple[Any, ...]] = []

        if is_comedy_remix:
            comedy_timeline = self.comedy_engine.plan_timeline(
                words_data=active_words,
                clip_start=start,
                clip_end=round(start + render_dur, 3),
                clip_spec=clip_spec,
                scene_cuts=scene_cuts
            )
            freeze_beat = comedy_timeline.freeze_beat
            if freeze_beat:
                f_start = freeze_beat.start_time
                f_dur = freeze_beat.duration
                sfx_path = self.sound_engine.get_cue_path(freeze_beat.sound_cue)
                if freeze_beat.reaction_asset_id:
                    rxn_meta = self.reaction_lib.get_reaction(freeze_beat.reaction_asset_id)
                    if rxn_meta and rxn_meta.file_path:
                        reaction_path = Path(rxn_meta.file_path)
                        if not reaction_path.is_absolute():
                            reaction_path = (config.BASE_DIR / reaction_path).resolve()
                        elif not reaction_path.exists():
                            reaction_path = (config.STORAGE_DIR / "assets" / "reactions" / reaction_path.name).resolve()

                # Shift downstream words forward by freeze duration
                active_words = self.comedy_engine.shift_words_for_timeline(
                    words_data=active_words,
                    freeze_start=start + f_start,
                    freeze_duration=f_dur
                )
                render_dur = round(render_dur + f_dur, 3)

                # Split keep_segs at f_start to insert freeze frame
                curr_t = 0.0
                freeze_placed = False
                for seg_s, seg_e in keep_segs:
                    seg_d = seg_e - seg_s
                    if not freeze_placed and curr_t <= f_start <= curr_t + seg_d:
                        offset = f_start - curr_t
                        split_src = round(seg_s + offset, 3)
                        if offset > 0.04:
                            render_items.append(('video', seg_s, split_src))
                        render_items.append(('freeze', split_src, f_dur))
                        if seg_e - split_src > 0.04:
                            render_items.append(('video', split_src, seg_e))
                        freeze_placed = True
                    else:
                        render_items.append(('video', seg_s, seg_e))
                    curr_t += seg_d
                if not freeze_placed:
                    render_items.insert(1, ('freeze', keep_segs[0][1], f_dur))
            else:
                render_items = [('video', s, e) for s, e in keep_segs]

        elif is_story_remix:
            storyboard = self.story_engine.create_storyboard(
                words_data=active_words,
                clip_start=start,
                clip_end=round(start + render_dur, 3),
                clip_spec=clip_spec,
                scene_cuts=scene_cuts
            )
            freeze_beat = storyboard.freeze_beat
            if freeze_beat:
                f_start = freeze_beat.start_time
                f_dur = freeze_beat.duration
                sfx_path = self.sound_engine.get_cue_path(freeze_beat.sound_cue)

                # Shift downstream words forward by freeze duration
                active_words = self.story_engine.shift_words_for_freeze(
                    words_data=active_words,
                    freeze_start=start + f_start,
                    freeze_duration=f_dur
                )
                render_dur = round(render_dur + f_dur, 3)

                # Split keep_segs at f_start to insert freeze frame
                curr_t = 0.0
                freeze_placed = False
                for seg_s, seg_e in keep_segs:
                    seg_d = seg_e - seg_s
                    if not freeze_placed and curr_t <= f_start <= curr_t + seg_d:
                        offset = f_start - curr_t
                        split_src = round(seg_s + offset, 3)
                        if offset > 0.04:
                            render_items.append(('video', seg_s, split_src))
                        render_items.append(('freeze', split_src, f_dur))
                        if seg_e - split_src > 0.04:
                            render_items.append(('video', split_src, seg_e))
                        freeze_placed = True
                    else:
                        render_items.append(('video', seg_s, seg_e))
                    curr_t += seg_d
                if not freeze_placed:
                    render_items.insert(1, ('freeze', keep_segs[0][1], f_dur))
            else:
                render_items = [('video', s, e) for s, e in keep_segs]
        else:
            render_items = [('video', s, e) for s, e in keep_segs]

        # 4. Context-aware narrative event detection & visual effect overlays
        clip_events = clip_spec.get("events")
        if clip_events is None:
            clip_events = self.event_detector.detect_events(
                words_data=active_words,
                clip_start=start,
                clip_end=round(start + render_dur, 3),
                clip_spec=clip_spec,
                scene_cuts=scene_cuts,
                intensity=edit_intensity
            )

        extra_filters = self.effect_engine.compile_video_filters(
            events=clip_events,
            preset=clip_preset
        )

        # 5. Generate ASS subtitles & animated visual overlays on the active timeline
        temp_ass = config.TEMP_DIR / f"{clip_id}.ass"
        ass_kwargs = {
            "words_data": active_words,
            "clip_start": start,
            "clip_end": round(start + render_dur, 3),
            "output_file": temp_ass,
            "preset": clip_preset,
            "style": clip_style,
            "font_name": clip_spec.get("font_name", font_name),
            "visual_events": clip_events
        }
        if is_comedy_remix and comedy_timeline:
            ass_kwargs["comedy_timeline"] = comedy_timeline
        elif is_story_remix and storyboard:
            ass_kwargs["storyboard"] = storyboard
        if font_size or clip_spec.get("font_size"):
            ass_kwargs["font_size"] = int(clip_spec.get("font_size", font_size))
        if accent_color or clip_spec.get("accent_color"):
            ass_kwargs["accent_color"] = clip_spec.get("accent_color", accent_color)
        else:
            palette = [
                "&H0000FFFF",  # Electric Yellow
                "&H00FFFF00",  # Vibrant Cyan
                "&H0000FF66",  # Neon Lime Green
                "&H000080FF",  # Warm Sunset Orange
                "&H00FF00FF",  # Electric Magenta
            ]
            import re
            m = re.search(r"(\d+)", clip_id)
            c_idx = (int(m.group(1)) - 1) if m else 0
            ass_kwargs["accent_color"] = palette[c_idx % len(palette)]
        if margin_v or clip_spec.get("margin_v"):
            ass_kwargs["margin_v"] = int(clip_spec.get("margin_v", margin_v))

        self.sub_gen.generate_ass(**ass_kwargs)

        # 6. Subject-aware framing analysis
        framing_info = self.framing_engine.analyze_clip_framing(
            video_path=source_video,
            start_time=start,
            duration=render_dur,
            framing_mode=clip_framing
        )

        # 7. Detect zoom emphasis intervals on the active timeline
        zoom_intervals = self.detect_zoom_intervals(
            words_data=active_words,
            clip_start=start,
            clip_end=round(start + render_dur, 3),
            preset=clip_preset,
            intensity=active_zoom_intensity
        )

        # 8. Build filtergraph
        src_probe = self.probe_video(source_video)

        if is_comedy_remix:
            # Phase 7.2 Full-Screen Comedy Remix Engine
            crop_w = framing_info.get("crop_w", 608)
            crop_h = framing_info.get("crop_h", 1080)
            crop_x = framing_info.get("crop_x", 420)

            # Ensure procedural animation & reaction assets exist
            speed_mov = str(self.anim_engine.speed_lines_path.resolve())
            burst_mov = str(self.anim_engine.radial_burst_path.resolve())
            halftone_path = self.anim_engine.asset_dir / "halftone.mov"
            if not halftone_path.exists():
                self.anim_engine.generate_halftone_texture(halftone_path)
            halftone_mov = str(halftone_path.resolve())

            # Prepare freeze segment if freeze_beat exists
            temp_freeze_mp4 = None
            if freeze_beat:
                f_dur = freeze_beat.duration
                split_src = start + freeze_beat.start_time
                freeze_item = next((it for it in render_items if it[0] == 'freeze'), None)
                if freeze_item:
                    split_src = freeze_item[1]

                temp_freeze_frame = config.TEMP_DIR / f"{clip_id}_comic_frame.png"
                temp_freeze_mp4 = config.TEMP_DIR / f"{clip_id}_comic_freeze.mp4"

                extract_cmd = [
                    config.FFMPEG_PATH, "-y",
                    "-ss", f"{split_src:.3f}",
                    "-i", str(source_video),
                    "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:0,scale=1080:1920,setsar=1",
                    "-vframes", "1",
                    str(temp_freeze_frame)
                ]
                subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

                import cv2
                frozen_bgr = cv2.imread(str(temp_freeze_frame))
                if frozen_bgr is not None:
                    self.anim_engine.create_comic_freeze_segment(
                        frozen_frame_bgr=frozen_bgr,
                        duration=f_dur,
                        output_mp4=temp_freeze_mp4,
                        sound_sfx_wav=sfx_path,
                        reaction_overlay_mov=reaction_path
                    )
                if temp_freeze_frame.exists():
                    temp_freeze_frame.unlink(missing_ok=True)

            filter_parts: List[str] = []
            concat_v: List[str] = []
            concat_a: List[str] = []
            cmd_inputs = ["-i", str(source_video)]

            freeze_input_idx = 1 if (temp_freeze_mp4 and temp_freeze_mp4.exists()) else None
            if freeze_input_idx:
                cmd_inputs.extend(["-i", str(temp_freeze_mp4)])

            for i, item in enumerate(render_items):
                itype = item[0]
                if itype == 'video':
                    seg_s, seg_e = item[1], item[2]
                    filter_parts.append(
                        f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:0,scale=1080:1920,setsar=1,"
                        f"trim=start={seg_s:.3f}:end={seg_e:.3f},setpts=PTS-STARTPTS[v{i}]"
                    )
                    if src_probe["has_audio"]:
                        filter_parts.append(f"[0:a]atrim=start={seg_s:.3f}:end={seg_e:.3f},asetpts=PTS-STARTPTS[a{i}]")
                elif itype == 'freeze':
                    if freeze_input_idx:
                        filter_parts.append(f"[{freeze_input_idx}:v]scale=1080:1920,setsar=1,setpts=PTS-STARTPTS[v{i}]")
                        filter_parts.append(
                            f"[{freeze_input_idx}:a]asetpts=PTS-STARTPTS,"
                            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a{i}]"
                        )
                    else:
                        src_t, f_dur = item[1], item[2]
                        filter_parts.append(
                            f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:0,scale=1080:1920,setsar=1,"
                            f"trim=start={src_t:.3f}:end={src_t+0.04:.3f},setpts=PTS-STARTPTS,"
                            f"tpad=stop_mode=clone:stop_duration={f_dur:.3f}[v{i}]"
                        )
                        filter_parts.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={f_dur:.3f}[a{i}]")
                concat_v.append(f"[v{i}]")
                if src_probe["has_audio"]:
                    concat_a.append(f"[a{i}]")

            if src_probe["has_audio"]:
                concat_inputs = "".join(f"{v}{a}" for v, a in zip(concat_v, concat_a))
                filter_parts.append(f"{concat_inputs}concat=n={len(render_items)}:v=1:a=1[vcat][acat]")
            else:
                filter_parts.append(f"{''.join(concat_v)}concat=n={len(render_items)}:v=1:a=0[vcat]")

            # Apply continuous visual transformations from comedy timeline
            current_v = "vcat"
            beats = comedy_timeline.beats if comedy_timeline else []
            for b_idx, beat in enumerate(beats):
                s, e = beat.start_time, beat.end_time
                if beat.beat_type == "speed_lines_shake":
                    next_v = f"v_shake_{b_idx}"
                    filter_parts.append(
                        f"[{current_v}]crop=w=in_w-36:h=in_h-36:"
                        f"x='(in_w-out_w)/2 + if(between(t\\,{s:.2f}\\,{e:.2f})\\,sin(t*45)*18\\,0)':"
                        f"y='(in_h-out_h)/2 + if(between(t\\,{s:.2f}\\,{e:.2f})\\,cos(t*35)*18\\,0)',"
                        f"scale=1080:1920[{next_v}]"
                    )
                    current_v = next_v
                    next_v = f"v_spd_{b_idx}"
                    spd_path = speed_mov.replace(":", r"\:")
                    filter_parts.append(f"movie={spd_path},setpts=PTS-STARTPTS+{s:.2f}/TB[spd_{b_idx}]")
                    filter_parts.append(
                        f"[{current_v}][spd_{b_idx}]overlay=0:0:enable='between(t\\,{s:.2f}\\,{e:.2f})'[{next_v}]"
                    )
                    current_v = next_v
                elif beat.beat_type == "radial_burst":
                    next_v = f"v_brst_{b_idx}"
                    brst_path = burst_mov.replace(":", r"\:")
                    filter_parts.append(f"movie={brst_path},setpts=PTS-STARTPTS+{s:.2f}/TB[brst_{b_idx}]")
                    filter_parts.append(
                        f"[{current_v}][brst_{b_idx}]overlay=0:0:enable='between(t\\,{s:.2f}\\,{e:.2f})'[{next_v}]"
                    )
                    current_v = next_v
                elif beat.beat_type == "impact_flash_reveal":
                    next_v = f"v_flash_{b_idx}"
                    filter_parts.append(
                        f"[{current_v}]drawbox=x=0:y=0:w=iw:h=ih:color=white@0.85:t=fill:enable='between(t\\,{s:.2f}\\,{min(e, s+0.10):.2f})'[{next_v}]"
                    )
                    current_v = next_v
                elif beat.beat_type == "confusion_desat":
                    next_v = f"v_desat_{b_idx}"
                    filter_parts.append(
                        f"[{current_v}]eq=saturation=0.35:brightness=-0.05:enable='between(t\\,{s:.2f}\\,{e:.2f})'[{next_v}]"
                    )
                    current_v = next_v
                elif beat.beat_type == "deadpan_snap":
                    next_v = f"v_snap_{b_idx}"
                    filter_parts.append(
                        f"[{current_v}]eq=saturation=1.5:contrast=1.15:enable='between(t\\,{s:.2f}\\,{e:.2f})'[{next_v}]"
                    )
                    current_v = next_v
                elif beat.beat_type == "comic_spotlight":
                    next_v = f"v_spot_{b_idx}"
                    filter_parts.append(
                        f"[{current_v}]vignette=angle=PI/3.2:enable='between(t\\,{s:.2f}\\,{e:.2f})'[{next_v}]"
                    )
                    current_v = next_v
                elif beat.beat_type == "cel_shaded_escalation":
                    next_v = f"v_cel_{b_idx}"
                    filter_parts.append(
                        f"[{current_v}]lutrgb='r=trunc(val/48)*48:g=trunc(val/48)*48:b=trunc(val/48)*48':enable='between(t\\,{s:.2f}\\,{e:.2f})',"
                        f"eq=saturation=2.2:contrast=1.3:enable='between(t\\,{s:.2f}\\,{e:.2f})'[{next_v}]"
                    )
                    current_v = next_v
                elif beat.beat_type == "halftone_texture":
                    next_v = f"v_ht_{b_idx}"
                    ht_path = halftone_mov.replace(":", r"\:")
                    filter_parts.append(f"movie={ht_path},setpts=PTS-STARTPTS+{s:.2f}/TB[ht_{b_idx}]")
                    filter_parts.append(
                        f"[{current_v}][ht_{b_idx}]overlay=0:0:enable='between(t\\,{s:.2f}\\,{e:.2f})'[{next_v}]"
                    )
                    current_v = next_v

            # Composite contextual animated emoji reactions
            emojis = comedy_timeline.emoji_reactions if comedy_timeline else []
            for em_idx, em in enumerate(emojis):
                temp_em_mov = config.TEMP_DIR / f"{clip_id}_emoji_{em_idx}.mov"
                self.emoji_engine.render_animated_emoji_mov(
                    reaction=em,
                    output_mov=temp_em_mov
                )
                em_path = str(temp_em_mov.resolve()).replace(":", r"\:")
                next_v = f"v_em_{em_idx}"
                filter_parts.append(f"movie={em_path},setpts=PTS-STARTPTS+{em.start_time:.2f}/TB[em_{em_idx}]")
                filter_parts.append(
                    f"[{current_v}][em_{em_idx}]overlay=0:0:enable='between(t\\,{em.start_time:.2f}\\,{em.start_time + em.duration:.2f})'[{next_v}]"
                )
                current_v = next_v

            # Burn ASS subtitles
            ass_clean = str(temp_ass.resolve()).replace(":", r"\:")
            filter_parts.append(f"[{current_v}]ass='{ass_clean}'[v]")

            # Audio normalization
            if src_probe["has_audio"]:
                if normalize_audio:
                    filter_parts.append("[acat]loudnorm=I=-16:TP=-1.5:LRA=11[aout]")
                else:
                    filter_parts.append("[acat]anull[aout]")

            full_filtergraph = ";".join(filter_parts)

            cmd = [config.FFMPEG_PATH, "-y"]
            cmd.extend(cmd_inputs)
            cmd.extend([
                "-filter_complex", full_filtergraph,
                "-map", "[v]"
            ])
            if src_probe["has_audio"]:
                cmd.extend(["-map", "[aout]"])

        elif is_story_remix:
            # Full Cartoon Remix Animation Pipeline
            crop_w = framing_info.get("crop_w", 608)
            crop_h = framing_info.get("crop_h", 1080)
            crop_x = framing_info.get("crop_x", 420)

            # Ensure procedural transparent animation assets exist
            speed_mov = str(self.anim_engine.speed_lines_path.resolve())
            burst_mov = str(self.anim_engine.radial_burst_path.resolve())

            # Prepare freeze segment if freeze_beat exists
            temp_freeze_mp4 = None
            if freeze_beat:
                f_dur = freeze_beat.duration
                split_src = start + freeze_beat.start_time
                freeze_item = next((it for it in render_items if it[0] == 'freeze'), None)
                if freeze_item:
                    split_src = freeze_item[1]

                temp_freeze_frame = config.TEMP_DIR / f"{clip_id}_comic_frame.png"
                temp_freeze_mp4 = config.TEMP_DIR / f"{clip_id}_comic_freeze.mp4"

                # Extract cropped frame at split_src
                extract_cmd = [
                    config.FFMPEG_PATH, "-y",
                    "-ss", f"{split_src:.3f}",
                    "-i", str(source_video),
                    "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:0,scale=1080:1920,setsar=1",
                    "-vframes", "1",
                    str(temp_freeze_frame)
                ]
                subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

                import cv2
                frozen_bgr = cv2.imread(str(temp_freeze_frame))
                if frozen_bgr is not None:
                    self.anim_engine.create_comic_freeze_segment(
                        frozen_frame_bgr=frozen_bgr,
                        duration=f_dur,
                        output_mp4=temp_freeze_mp4,
                        sound_sfx_wav=sfx_path
                    )
                if temp_freeze_frame.exists():
                    temp_freeze_frame.unlink(missing_ok=True)

            # Build filter parts for remix
            filter_parts: List[str] = []
            concat_v: List[str] = []
            concat_a: List[str] = []
            cmd_inputs = ["-i", str(source_video)]

            freeze_input_idx = 1 if (temp_freeze_mp4 and temp_freeze_mp4.exists()) else None
            if freeze_input_idx:
                cmd_inputs.extend(["-i", str(temp_freeze_mp4)])

            for i, item in enumerate(render_items):
                itype = item[0]
                if itype == 'video':
                    seg_s, seg_e = item[1], item[2]
                    filter_parts.append(
                        f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:0,scale=1080:1920,setsar=1,"
                        f"trim=start={seg_s:.3f}:end={seg_e:.3f},setpts=PTS-STARTPTS[v{i}]"
                    )
                    if src_probe["has_audio"]:
                        filter_parts.append(f"[0:a]atrim=start={seg_s:.3f}:end={seg_e:.3f},asetpts=PTS-STARTPTS[a{i}]")
                elif itype == 'freeze':
                    if freeze_input_idx:
                        filter_parts.append(f"[{freeze_input_idx}:v]scale=1080:1920,setsar=1,setpts=PTS-STARTPTS[v{i}]")
                        filter_parts.append(
                            f"[{freeze_input_idx}:a]asetpts=PTS-STARTPTS,"
                            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a{i}]"
                        )
                    else:
                        src_t, f_dur = item[1], item[2]
                        filter_parts.append(
                            f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:0,scale=1080:1920,setsar=1,"
                            f"trim=start={src_t:.3f}:end={src_t+0.04:.3f},setpts=PTS-STARTPTS,"
                            f"tpad=stop_mode=clone:stop_duration={f_dur:.3f}[v{i}]"
                        )
                        filter_parts.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={f_dur:.3f}[a{i}]")
                concat_v.append(f"[v{i}]")
                if src_probe["has_audio"]:
                    concat_a.append(f"[a{i}]")

            if src_probe["has_audio"]:
                concat_inputs = "".join(f"{v}{a}" for v, a in zip(concat_v, concat_a))
                filter_parts.append(f"{concat_inputs}concat=n={len(render_items)}:v=1:a=1[vcat][acat]")
            else:
                filter_parts.append(f"{''.join(concat_v)}concat=n={len(render_items)}:v=1:a=0[vcat]")

            # Now apply linear video animations from storyboard.video_animations!
            current_v = "vcat"
            anim_beats = storyboard.video_animations if storyboard else []
            for a_idx, anim in enumerate(anim_beats):
                s, e = anim.start_time, anim.end_time
                if anim.anim_type == "screen_shake":
                    next_v = f"v_shake_{a_idx}"
                    filter_parts.append(
                        f"[{current_v}]crop=w=in_w-36:h=in_h-36:"
                        f"x='(in_w-out_w)/2 + if(between(t\\,{s:.2f}\\,{e:.2f})\\,sin(t*45)*16\\,0)':"
                        f"y='(in_h-out_h)/2 + if(between(t\\,{s:.2f}\\,{e:.2f})\\,cos(t*35)*16\\,0)',"
                        f"scale=1080:1920[{next_v}]"
                    )
                    current_v = next_v
                elif anim.anim_type == "speed_lines":
                    next_v = f"v_spd_{a_idx}"
                    spd_path = speed_mov.replace(":", r"\:")
                    filter_parts.append(f"movie={spd_path},setpts=PTS-STARTPTS+{s:.2f}/TB[spd_{a_idx}]")
                    filter_parts.append(
                        f"[{current_v}][spd_{a_idx}]overlay=0:0:enable='between(t\\,{s:.2f}\\,{e:.2f})'[{next_v}]"
                    )
                    current_v = next_v
                elif anim.anim_type == "radial_burst":
                    next_v = f"v_brst_{a_idx}"
                    brst_path = burst_mov.replace(":", r"\:")
                    filter_parts.append(f"movie={brst_path},setpts=PTS-STARTPTS+{s:.2f}/TB[brst_{a_idx}]")
                    filter_parts.append(
                        f"[{current_v}][brst_{a_idx}]overlay=0:0:enable='between(t\\,{s:.2f}\\,{e:.2f})'[{next_v}]"
                    )
                    current_v = next_v
                elif anim.anim_type == "cel_shaded_video":
                    next_v = f"v_cel_{a_idx}"
                    filter_parts.append(
                        f"[{current_v}]lutrgb='r=trunc(val/48)*48:g=trunc(val/48)*48:b=trunc(val/48)*48':enable='between(t\\,{s:.2f}\\,{e:.2f})',"
                        f"eq=saturation=2.2:contrast=1.3:enable='between(t\\,{s:.2f}\\,{e:.2f})'[{next_v}]"
                    )
                    current_v = next_v

            # Burn ASS subtitles
            ass_clean = str(temp_ass.resolve()).replace(":", r"\:")
            filter_parts.append(f"[{current_v}]ass='{ass_clean}'[v]")

            # Audio normalization
            if src_probe["has_audio"]:
                if normalize_audio:
                    filter_parts.append("[acat]loudnorm=I=-16:TP=-1.5:LRA=11[aout]")
                else:
                    filter_parts.append("[acat]anull[aout]")

            full_filtergraph = ";".join(filter_parts)

            cmd = [config.FFMPEG_PATH, "-y"]
            cmd.extend(cmd_inputs)
            cmd.extend([
                "-filter_complex", full_filtergraph,
                "-map", "[v]"
            ])
            if src_probe["has_audio"]:
                cmd.extend(["-map", "[aout]"])

        elif len(render_items) > 1 or freeze_beat:
            # Multi-segment trim and concat filtergraph (supports dead-air trim and freeze frames)
            filter_parts: List[str] = []
            concat_v: List[str] = []
            concat_a: List[str] = []

            for i, item in enumerate(render_items):
                itype = item[0]
                if itype == 'video':
                    seg_s, seg_e = item[1], item[2]
                    filter_parts.append(f"[0:v]trim=start={seg_s:.3f}:end={seg_e:.3f},setpts=PTS-STARTPTS[v{i}]")
                    if src_probe["has_audio"]:
                        filter_parts.append(f"[0:a]atrim=start={seg_s:.3f}:end={seg_e:.3f},asetpts=PTS-STARTPTS[a{i}]")
                elif itype == 'freeze':
                    src_t, f_dur = item[1], item[2]
                    filter_parts.append(
                        f"[0:v]trim=start={src_t:.3f}:end={src_t+0.04:.3f},setpts=PTS-STARTPTS,"
                        f"tpad=stop_mode=clone:stop_duration={f_dur:.3f},"
                        f"vignette=PI/4:enable='between(t\\,0\\,{f_dur:.3f})'[v{i}]"
                    )
                    if sfx_path and src_probe["has_audio"]:
                        filter_parts.append(
                            f"[1:a]apad=pad_dur={f_dur:.3f},atrim=start=0:end={f_dur:.3f},"
                            f"asetpts=PTS-STARTPTS,volume=0.9[a{i}]"
                        )
                    elif src_probe["has_audio"]:
                        filter_parts.append(
                            f"anullsrc=r=48000:cl=stereo,atrim=duration={f_dur:.3f}[a{i}]"
                        )

                concat_v.append(f"[v{i}]")
                if src_probe["has_audio"]:
                    concat_a.append(f"[a{i}]")

            if src_probe["has_audio"]:
                concat_inputs = "".join(f"{v}{a}" for v, a in zip(concat_v, concat_a))
                filter_parts.append(f"{concat_inputs}concat=n={len(render_items)}:v=1:a=1[vcat][acat]")
            else:
                filter_parts.append(f"{''.join(concat_v)}concat=n={len(render_items)}:v=1:a=0[vcat]")

            # Apply visual framing and subtitle burning to [vcat]
            v_chain = self.framing_engine.build_filtergraph(
                framing_info,
                temp_ass,
                zoom_intervals=zoom_intervals,
                input_v_label="vcat",
                extra_filters=extra_filters
            )
            filter_parts.append(v_chain)

            # Audio normalization chain
            if src_probe["has_audio"]:
                if normalize_audio:
                    filter_parts.append("[acat]loudnorm=I=-16:TP=-1.5:LRA=11[aout]")
                else:
                    filter_parts.append("[acat]anull[aout]")

            full_filtergraph = ";".join(filter_parts)

            cmd = [
                config.FFMPEG_PATH,
                "-y",
                "-i", str(source_video)
            ]
            if sfx_path and freeze_beat:
                cmd.extend(["-i", str(sfx_path)])

            cmd.extend([
                "-filter_complex", full_filtergraph,
                "-map", "[v]"
            ])
            if src_probe["has_audio"]:
                cmd.extend(["-map", "[aout]"])

        else:
            # Single-segment fast path
            filtergraph = self.framing_engine.build_filtergraph(
                framing_info,
                temp_ass,
                zoom_intervals=zoom_intervals,
                input_v_label="0:v",
                extra_filters=extra_filters
            )

            cmd = [
                config.FFMPEG_PATH,
                "-y",
                "-ss", str(start),
                "-t", str(render_dur),
                "-i", str(source_video),
                "-filter_complex", filtergraph,
                "-map", "[v]"
            ]

            if src_probe["has_audio"]:
                cmd.extend(["-map", "0:a"])
                if normalize_audio:
                    cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])

        if src_probe["has_audio"]:
            cmd.extend([
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000"
            ])

        # Phase 6 Output Protection: Write to temporary file, probe-validate, then atomic swap
        temp_out_file = out_file.with_suffix(".tmp.mp4")
        temp_out_file.unlink(missing_ok=True)

        cmd.extend([
            "-c:v", "libx264",
            "-r", "24000/1001",
            "-preset", "fast",
            "-crf", "19",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(temp_out_file)
        ])

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            temp_out_file.unlink(missing_ok=True)
            err_msg = result.stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f"FFmpeg render failure: {err_msg}")

        if not temp_out_file.exists() or temp_out_file.stat().st_size < 5000:
            temp_out_file.unlink(missing_ok=True)
            raise RuntimeError(f"Output clip {temp_out_file} is missing or corrupted.")

        # Probe and validate the rendered temporary file before replacing existing output
        out_probe = self.probe_video(temp_out_file)
        if out_probe["width"] != 1080 or out_probe["height"] != 1920:
            temp_out_file.unlink(missing_ok=True)
            raise RuntimeError(
                f"Validation failed: expected 1080x1920 vertical format, got {out_probe['width']}x{out_probe['height']}"
            )

        # Atomic replacement: safe against crashes or partial writes
        temp_out_file.replace(out_file)
        filesize_mb = round(out_file.stat().st_size / (1024 * 1024), 2)

        return {
            "id": clip_id,
            "filename": out_filename,
            "filepath": str(out_file.resolve()),
            "duration": render_dur,
            "original_duration": duration,
            "dead_air_trimmed": pacing_res["dead_air_trimmed"],
            "trim_count": pacing_res["trim_count"],
            "filesize_mb": filesize_mb,
            "resolution": f"{out_probe['width']}x{out_probe['height']}",
            "preset": clip_preset,
            "editing_intensity": edit_intensity,
            "silence_trim": active_silence_trim,
            "zoom_intensity": active_zoom_intensity,
            "framing_layout": framing_info.get("layout"),
            "framing_reason": framing_info.get("reason"),
            "zoom_intervals": zoom_intervals,
            "scene_cuts": scene_cuts,
            "hook": clip_spec.get("hook", ""),
            "hook_type": clip_spec.get("hook_type", "declarative"),
            "reason": clip_spec.get("reason", ""),
            "events": [e.to_dict() if hasattr(e, "to_dict") else e for e in clip_events],
            "storyboard": storyboard.to_dict() if storyboard else None,
            "comedy_timeline": comedy_timeline.to_dict() if comedy_timeline else None,
            "emoji_reactions": [e.to_dict() for e in comedy_timeline.emoji_reactions] if comedy_timeline else [],
            "originality": storyboard.originality.to_dict() if storyboard else None,
            "freeze_beat": freeze_beat.to_dict() if freeze_beat else None,
            "sound_cues": storyboard.sound_cues if storyboard else []
        }
