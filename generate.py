#!/usr/bin/env python3
"""
AI Shorts Generator - Primary Entry Point.
Takes a YouTube URL and/or prompt, selects the 5 best narrative moments,
and renders vertical 9:16 Shorts with original footage, smart crop, dynamic zooms,
animated subtitles, contextual animated emojis, sound cues, and loudness normalization.

Every run creates a brand new dedicated output folder inside output/ containing all 5 clips.
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.batch import BatchProcessor, parse_batch_request, DEFAULT_COUNT, DEFAULT_MAX_DURATION


def main():
    parser = argparse.ArgumentParser(
        description="Local AI Shorts Generator: transform any YouTube video into 5 high-energy Shorts."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="YouTube video URL or natural language prompt containing a URL."
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Explicit YouTube video URL."
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Creative prompt / styling instruction (e.g. 'Make 5 funny clips with high energy')."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of shorts to generate (default: 5)."
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum duration per clip in seconds (default: 30s)."
    )
    parser.add_argument(
        "--preset",
        default="full_comedy_remix",
        choices=["full_comedy_remix", "vibrant_pro", "story_remix", "high_energy", "clean_pro"],
        help="Visual editing preset (default: full_comedy_remix)."
    )
    parser.add_argument(
        "--editing-intensity",
        default="dynamic",
        choices=["subtle", "balanced", "dynamic"],
        help="Editing pacing and zoom intensity (default: dynamic)."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional custom output directory. If omitted, a new unique folder output/run_<timestamp>_<id> is created."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-rendering of clips even if previously cached."
    )

    args = parser.parse_args()

    # Determine input URL and prompt
    raw_input = args.target or args.url or ""
    settings = {}
    if args.prompt:
        settings = parse_batch_request(args.prompt)
    elif raw_input:
        settings = parse_batch_request(raw_input)

    # CLI explicit overrides
    if args.url:
        settings["url"] = args.url
    elif raw_input.startswith("http") or "youtube.com" in raw_input or "youtu.be" in raw_input:
        settings["url"] = raw_input

    if args.count:
        settings["count"] = args.count
    if args.max_duration:
        settings["max_duration"] = args.max_duration
    if args.preset:
        settings["preset"] = args.preset
    if args.editing_intensity:
        settings["editing_intensity"] = args.editing_intensity

    if not settings.get("url") and not settings.get("video_path"):
        print("\n❌ Error: Please provide a valid YouTube URL.")
        print("Usage:")
        print("  python generate.py \"https://www.youtube.com/watch?v=plN7JMbadRg\"")
        print("  python generate.py --url \"https://www.youtube.com/watch?v=plN7JMbadRg\" --count 5")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🎬 LOCAL AI SHORTS GENERATOR")
    print("=" * 70)
    print(f" Source URL    : {settings.get('url')}")
    print(f" Target Clips  : {settings.get('count', 5)}")
    print(f" Max Duration  : {settings.get('max_duration', 30.0)}s")
    print(f" Preset        : {settings.get('preset', 'full_comedy_remix')}")
    print(f" Footing Mode  : Original High-Resolution Footage (9:16 Smart Crop)")
    print(f" Enhancements  : Dynamic Zooms, Animated Subtitles, Emojis, Audio Cues")
    print("=" * 70 + "\n")

    processor = BatchProcessor()
    out_dir = Path(args.output_dir) if args.output_dir else None

    try:
        manifest = processor.process_batch(
            settings=settings,
            output_dir=out_dir,
            force_rerender=args.force
        )

        run_folder = manifest.get("output_dir", "")
        rendered = manifest.get("rendered_shorts", [])

        print("\n" + "=" * 70)
        print("✅ BATCH GENERATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"📁 Output Folder : {run_folder}")
        print(f"🎞️ Clips Rendered: {len(rendered)} / {settings.get('count', 5)}")
        print(f"📄 Manifest File : {manifest.get('manifest_path')}")
        if manifest.get("contact_sheet"):
            print(f"🖼️ Contact Sheet : {manifest.get('contact_sheet')}")
        print("-" * 70)

        for i, clip in enumerate(rendered, 1):
            cid = clip.get("id", f"short_{i}")
            dur = clip.get("duration", 0.0)
            hook = clip.get("hook", "")
            fpath = clip.get("filepath", "")
            score = clip.get("score", 75)
            print(f"  [{i}] {cid} ({dur:.1f}s, score={score})")
            if hook:
                print(f"      Hook: \"{hook}\"")
            print(f"      Path: file://{fpath}")

        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error during batch generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
