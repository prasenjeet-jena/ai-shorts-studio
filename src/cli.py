import argparse
import json
import sys
from pathlib import Path

from src.ingestion import IngestionManager, validate_youtube_url
from src.rendering import VideoRenderer
from src.config import config

def cmd_ingest(args):
    url = args.url.strip()
    if not validate_youtube_url(url):
        print(json.dumps({"status": "error", "message": f"Invalid YouTube URL: {url}"}))
        sys.exit(1)

    manager = IngestionManager()
    out_dir = Path(args.output_dir) if args.output_dir else None

    try:
        video_path, audio_path, metadata = manager.download(url, output_dir=out_dir)
        result = {
            "status": "success",
            "video_id": metadata["video_id"],
            "title": metadata["title"],
            "author": metadata["author"],
            "duration": metadata["duration"],
            "video_path": str(video_path.resolve()),
            "audio_path": str(audio_path.resolve()),
            "thumbnail": metadata.get("thumbnail")
        }
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

def cmd_transcribe(args):
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(json.dumps({"status": "error", "message": f"Audio file not found: {audio_path}"}))
        sys.exit(1)

    from src.transcription import SpeechTranscriber
    transcriber = SpeechTranscriber(model_size=args.model)
    try:
        result = transcriber.transcribe(
            audio_path=audio_path,
            video_id=args.video_id,
            block_duration=args.block_duration
        )
        result["status"] = "success"
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

def cmd_render(args):
    video_path = Path(args.video)
    words_path = Path(args.words)
    clips_path = Path(args.clips)
    out_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR

    if not video_path.exists():
        print(json.dumps({"status": "error", "message": f"Source video not found: {video_path}"}))
        sys.exit(1)
    if not words_path.exists():
        print(json.dumps({"status": "error", "message": f"Words file not found: {words_path}"}))
        sys.exit(1)
    if not clips_path.exists():
        print(json.dumps({"status": "error", "message": f"Clips spec file not found: {clips_path}"}))
        sys.exit(1)

    try:
        with open(words_path, "r", encoding="utf-8") as f:
            words_data = json.load(f)
        if isinstance(words_data, dict) and "data" in words_data:
            words_data = words_data["data"]
        with open(clips_path, "r", encoding="utf-8") as f:
            clips_spec = json.load(f)

        if isinstance(clips_spec, dict):
            clips_spec = [clips_spec]

        renderer = VideoRenderer()
        rendered_clips = []

        preset = args.preset or args.style or "vibrant_pro"
        for clip in clips_spec:
            clip_res = renderer.render_clip(
                source_video=video_path,
                words_data=words_data,
                clip_spec=clip,
                output_dir=out_dir,
                preset=preset,
                style=args.style,
                framing=args.framing,
                font_name=args.font_name,
                font_size=args.font_size,
                accent_color=args.accent_color,
                margin_v=args.margin_v,
                normalize_audio=not args.no_audio_norm,
                editing_intensity=args.editing_intensity,
                silence_trim=args.silence_trim,
                zoom_intensity=args.zoom_intensity,
                scene_detect=args.scene_detect
            )
            rendered_clips.append(clip_res)

        print(json.dumps({
            "status": "success",
            "count": len(rendered_clips),
            "rendered_clips": rendered_clips
        }, indent=2))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

def cmd_validate(args):
    file_path = Path(args.file)
    if not file_path.exists():
        print(json.dumps({"status": "error", "message": f"File not found: {file_path}"}))
        sys.exit(1)

    renderer = VideoRenderer()
    try:
        probe = renderer.probe_video(file_path)
        is_vertical = probe["height"] > probe["width"]
        print(json.dumps({
            "status": "success",
            "file": str(file_path.resolve()),
            "width": probe["width"],
            "height": probe["height"],
            "is_vertical_9_16": (probe["width"] == 1080 and probe["height"] == 1920) or is_vertical,
            "codec": probe["codec"],
            "filesize_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
        }, indent=2))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

def cmd_propose(args):
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(json.dumps({"status": "error", "message": f"Summary file not found: {summary_path}"}))
        sys.exit(1)

    from src.selector import CandidateSelector
    from src.scene import SceneDetector
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_blocks = json.load(f)

        scene_cuts = None
        if getattr(args, "video", None) and Path(args.video).exists():
            detector = SceneDetector()
            # detect cuts across first 120s
            scene_cuts = detector.detect_scene_cuts(Path(args.video), 0.0, 120.0)

        selector = CandidateSelector()
        proposals = selector.propose_candidates(
            summary_blocks=summary_blocks,
            max_duration=args.max_duration,
            count=args.count,
            scene_cuts=scene_cuts
        )
        print(json.dumps({
            "status": "success",
            "count": len(proposals),
            "candidates": proposals
        }, indent=2))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

def cmd_batch(args):
    from src.batch import BatchProcessor, parse_batch_request
    
    settings = {}
    if getattr(args, "prompt", None):
        settings = parse_batch_request(args.prompt)

    if getattr(args, "url", None):
        settings["url"] = args.url
    if getattr(args, "video", None):
        settings["video_path"] = args.video
    if getattr(args, "count", None) is not None:
        settings["count"] = args.count
    if getattr(args, "max_duration", None) is not None:
        settings["max_duration"] = args.max_duration
    if getattr(args, "editing_intensity", None):
        settings["editing_intensity"] = args.editing_intensity
    if getattr(args, "preset", None):
        settings["preset"] = args.preset

    min_separation = getattr(args, "min_separation", 0.0)
    out_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None

    agent_candidates = None
    if getattr(args, "candidates", None):
        cand_arg = args.candidates.strip()
        if cand_arg.startswith("["):
            agent_candidates = json.loads(cand_arg)
        else:
            cand_path = Path(cand_arg)
            if cand_path.exists():
                with open(cand_path, "r", encoding="utf-8") as f:
                    agent_candidates = json.load(f)

    try:
        processor = BatchProcessor()
        manifest = processor.process_batch(
            settings=settings,
            output_dir=out_dir,
            min_separation=min_separation,
            agent_candidates=agent_candidates,
            run_id=getattr(args, "run_id", None),
            force_rerender=getattr(args, "force_rerender", False),
            skip_preflight=getattr(args, "skip_preflight", False)
        )
        print(json.dumps(manifest, indent=2))
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

def cmd_resume(args):
    from src.batch import BatchProcessor
    try:
        processor = BatchProcessor()
        manifest = processor.resume_batch(
            run_id=args.run_id,
            force_rerender=args.force
        )
        print(json.dumps(manifest, indent=2))
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

def cmd_preflight(args):
    from src.preflight import run_preflight, PreflightError
    try:
        res = run_preflight(min_disk_gb=args.min_disk_gb, require_node=args.require_node)
        print(json.dumps(res, indent=2))
    except PreflightError as pe:
        print(json.dumps({"ready": False, "message": str(pe), "critical_failed": pe.failed_checks}, indent=2))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"ready": False, "message": str(e)}, indent=2))
        sys.exit(1)

def cmd_runs(args):
    from src.run_manager import run_manager
    if args.action == "list":
        runs = run_manager.list_runs()
        print(json.dumps({"status": "success", "count": len(runs), "runs": runs}, indent=2))
    elif args.action == "status":
        state = run_manager.load_run(args.run_id)
        if not state:
            print(json.dumps({"status": "error", "message": f"Run '{args.run_id}' not found."}))
            sys.exit(1)
        print(json.dumps(state, indent=2))

def cmd_cache(args):
    from src.cache import cache_manager
    if args.action == "clean":
        cleared = cache_manager.invalidate(video_id=args.video_id)
        print(json.dumps({"status": "success", "cleared_entries": cleared}))

def main():
    parser = argparse.ArgumentParser(description="Local AI Shorts Generator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Batch
    p_batch = subparsers.add_parser("batch", help="Intelligent batch generation of YouTube Shorts")
    p_batch.add_argument("--prompt", default=None, help="Natural language request string")
    p_batch.add_argument("--url", default=None, help="YouTube video URL")
    p_batch.add_argument("--video", default=None, help="Local source video file")
    p_batch.add_argument("--count", type=int, default=None, help="Number of clips (3, 5, or 10)")
    p_batch.add_argument("--max-duration", type=float, default=None, help="Maximum duration in seconds (15, 30, 45, or 60)")
    p_batch.add_argument("--editing-intensity", default=None, choices=["subtle", "balanced", "dynamic"], help="Editing intensity")
    p_batch.add_argument("--preset", default=None, choices=["clean_pro", "vibrant_pro", "high_energy", "story_remix", "full_comedy_remix"], help="Visual style preset")
    p_batch.add_argument("--min-separation", type=float, default=0.0, help="Minimum separation between clips in seconds")
    p_batch.add_argument("--candidates", default=None, help="Path to JSON file or JSON string of candidate specifications")
    p_batch.add_argument("--run-id", default=None, help="Custom run identifier")
    p_batch.add_argument("--force-rerender", action="store_true", help="Force re-rendering of existing clips")
    p_batch.add_argument("--skip-preflight", action="store_true", help="Bypass preflight health checks")
    p_batch.add_argument("--output-dir", default=None, help="Output directory for generated batch clips")
    p_batch.set_defaults(func=cmd_batch)

    # Resume
    p_resume = subparsers.add_parser("resume", help="Resume an interrupted batch run")
    p_resume.add_argument("--run-id", required=True, help="Run ID to resume")
    p_resume.add_argument("--force", action="store_true", help="Force re-render of already completed clips")
    p_resume.set_defaults(func=cmd_resume)

    # Preflight
    p_pref = subparsers.add_parser("preflight", help="Run local system health and dependency checks")
    p_pref.add_argument("--min-disk-gb", type=float, default=2.0, help="Minimum free disk space in GB")
    p_pref.add_argument("--require-node", action="store_true", help="Require Node.js runtime check to pass")
    p_pref.set_defaults(func=cmd_preflight)

    # Runs management
    p_runs = subparsers.add_parser("runs", help="Inspect and manage persistent runs")
    p_runs_sub = p_runs.add_subparsers(dest="action", required=True)
    p_runs_list = p_runs_sub.add_parser("list", help="List all recorded runs")
    p_runs_status = p_runs_sub.add_parser("status", help="Get detailed status of a specific run")
    p_runs_status.add_argument("--run-id", required=True, help="Target run ID")
    p_runs.set_defaults(func=cmd_runs)

    # Cache management
    p_cache = subparsers.add_parser("cache", help="Manage local caching")
    p_cache_sub = p_cache.add_subparsers(dest="action", required=True)
    p_cache_clean = p_cache_sub.add_parser("clean", help="Safely invalidate cache files")
    p_cache_clean.add_argument("--video-id", default=None, help="Optional video ID to invalidate")
    p_cache.set_defaults(func=cmd_cache)

    # Ingest
    p_ingest = subparsers.add_parser("ingest", help="Download YouTube video and extract audio")
    p_ingest.add_argument("--url", required=True, help="YouTube video URL")
    p_ingest.add_argument("--output-dir", default=None, help="Target storage directory")
    p_ingest.set_defaults(func=cmd_ingest)

    # Transcribe
    p_trans = subparsers.add_parser("transcribe", help="Transcribe speech with Faster-Whisper")
    p_trans.add_argument("--audio", required=True, help="Path to 16kHz WAV audio")
    p_trans.add_argument("--video-id", required=True, help="Video ID for caching")
    p_trans.add_argument("--model", default=None, help="Whisper model size (base.en, small, medium)")
    p_trans.add_argument("--block-duration", type=float, default=30.0, help="Semantic block target duration")
    p_trans.set_defaults(func=cmd_transcribe)

    # Propose
    p_prop = subparsers.add_parser("propose", help="Propose non-overlapping candidate moments")
    p_prop.add_argument("--summary", required=True, help="Path to summary.json")
    p_prop.add_argument("--video", default=None, help="Optional source video for scene cut boundary snapping")
    p_prop.add_argument("--max-duration", type=float, default=30.0, help="Max clip duration")
    p_prop.add_argument("--count", type=int, default=5, help="Number of clips")
    p_prop.set_defaults(func=cmd_propose)

    # Render
    p_render = subparsers.add_parser("render", help="Render vertical 9:16 Shorts with FFmpeg")
    p_render.add_argument("--video", required=True, help="Source MP4 video file")
    p_render.add_argument("--words", required=True, help="Path to words.json")
    p_render.add_argument("--clips", required=True, help="Path to selected clips JSON spec")
    p_render.add_argument("--preset", default="vibrant_pro", choices=["clean_pro", "vibrant_pro", "high_energy", "story_remix", "full_comedy_remix"], help="Visual editing preset")
    p_render.add_argument("--style", default=None, choices=["dynamic", "clean", "cinematic"], help="Legacy style alias")
    p_render.add_argument("--editing-intensity", default="balanced", choices=["subtle", "balanced", "dynamic"], help="Phase 4 editing pacing intensity")
    p_render.add_argument("--silence-trim", default="conservative", choices=["off", "conservative", "moderate"], help="Dead-air trimming sensitivity")
    p_render.add_argument("--zoom-intensity", default="selective", choices=["off", "selective", "dynamic"], help="Dynamic punch-in zoom frequency")
    p_render.add_argument("--scene-detect", action=argparse.BooleanOptionalAction, default=True, help="Enable scene change detection")
    p_render.add_argument("--framing", default="auto", choices=["auto", "crop", "blur"], help="Vertical framing strategy")
    p_render.add_argument("--font-name", default="Arial", help="Caption font face")
    p_render.add_argument("--font-size", type=int, default=None, help="Caption font size in pt")
    p_render.add_argument("--accent-color", default=None, help="Word highlight accent color (&H0000FFFF)")
    p_render.add_argument("--margin-v", type=int, default=None, help="Vertical safe margin in px")
    p_render.add_argument("--no-audio-norm", action="store_true", help="Disable EBU R128 loudness normalization")
    p_render.add_argument("--output-dir", default=None, help="Output clips directory")
    p_render.set_defaults(func=cmd_render)

    # Validate
    p_val = subparsers.add_parser("validate", help="Validate output MP4 video properties")
    p_val.add_argument("--file", required=True, help="Path to rendered MP4")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
