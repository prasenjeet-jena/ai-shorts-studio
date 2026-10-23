import unittest
import json
import numpy as np
from pathlib import Path

from src.reactions import ReactionLibrary, ReactionMetadata
from src.comedy_timeline import FullComedyRemixEngine, FullComedyTimeline, ComedyBeat
from src.subtitles import SubtitleGenerator
from src.anim import AnimationEngine
from src.config import config


class TestPhase72ComedyRemix(unittest.TestCase):
    def setUp(self):
        self.reaction_lib = ReactionLibrary()
        self.comedy_engine = FullComedyRemixEngine()
        self.sub_gen = SubtitleGenerator()
        self.anim_engine = AnimationEngine()

    def test_reaction_library_registry_and_metadata(self):
        """Verifies that all procedural reaction assets are registered with CC0/FOSS metadata and exist on disk."""
        reactions = self.reaction_lib.list_reactions()
        self.assertGreaterEqual(len(reactions), 4)

        expected_ids = [
            "reaction_buffering_brain",
            "reaction_shock_eyes",
            "reaction_celebration_confetti",
            "reaction_dramatic_iris",
            "reaction_facepalm",
            "reaction_awkward_silence",
        ]
        for rxn_id in expected_ids:
            rxn = self.reaction_lib.get_reaction(rxn_id)
            self.assertIsNotNone(rxn, f"Missing reaction: {rxn_id}")
            self.assertTrue(Path(rxn.file_path).exists(), f"File does not exist: {rxn.file_path}")
            self.assertIn("CC0", rxn.license)
            self.assertGreater(rxn.duration, 0.5)

    def test_full_comedy_remix_timeline_planning(self):
        """Verifies that FullComedyRemixEngine plans a timeline meeting the 1-3s pacing target."""
        words_data = [
            {"word": "I", "start": 0.0, "end": 0.3},
            {"word": "just", "start": 0.3, "end": 0.5},
            {"word": "bought", "start": 0.5, "end": 0.8},
            {"word": "this", "start": 0.8, "end": 1.0},
            {"word": "entire", "start": 1.0, "end": 1.3},
            {"word": "grocery", "start": 1.3, "end": 1.6},
            {"word": "store", "start": 1.6, "end": 2.0},
            {"word": "win", "start": 6.0, "end": 6.5},
            {"word": "one", "start": 6.5, "end": 6.8},
            {"word": "million", "start": 6.8, "end": 7.3},
            {"word": "dollars", "start": 7.3, "end": 7.8},
            {"word": "Oh", "start": 8.0, "end": 8.3},
            {"word": "my", "start": 8.3, "end": 8.5},
            {"word": "god", "start": 8.5, "end": 8.9},
            {"word": "Everything", "start": 11.5, "end": 12.2},
            {"word": "Millions", "start": 17.5, "end": 18.2},
            {"word": "calories", "start": 18.2, "end": 19.0},
            {"word": "impossible", "start": 21.0, "end": 22.0},
            {"word": "Wow", "start": 23.0, "end": 23.8}
        ]
        clip_spec = {
            "id": "test_remix_clip",
            "start": 0.0,
            "end": 23.85,
            "hook": "I just bought this grocery store"
        }

        timeline = self.comedy_engine.plan_timeline(
            words_data=words_data,
            clip_start=0.0,
            clip_end=23.85,
            clip_spec=clip_spec
        )

        self.assertIsInstance(timeline, FullComedyTimeline)
        self.assertGreaterEqual(len(timeline.beats), 10)
        self.assertIsNotNone(timeline.freeze_beat)
        self.assertEqual(timeline.freeze_beat.reaction_asset_id, "reaction_buffering_brain")

        # Pacing target: 1.0s to 3.0s average interval
        pacing = timeline.pacing_stats
        self.assertGreaterEqual(pacing["average_interval_seconds"], 1.0)
        self.assertLessEqual(pacing["average_interval_seconds"], 3.0)
        self.assertEqual(pacing["pacing_compliance"], "PASS")

    def test_caption_synchronization_shift(self):
        """Verifies that downstream words are accurately shifted forward by freeze duration."""
        words_data = [
            {"word": "BEFORE", "start": 5.0, "end": 5.5},
            {"word": "DURING", "start": 8.8, "end": 9.0},
            {"word": "AFTER", "start": 10.5, "end": 11.0}
        ]
        freeze_start = 9.0
        freeze_duration = 1.2

        shifted = self.comedy_engine.shift_words_for_timeline(
            words_data=words_data,
            freeze_start=freeze_start,
            freeze_duration=freeze_duration
        )

        self.assertEqual(shifted[0]["start"], 5.0)
        self.assertEqual(shifted[0]["end"], 5.5)
        self.assertAlmostEqual(shifted[2]["start"], 11.7, places=2)
        self.assertAlmostEqual(shifted[2]["end"], 12.2, places=2)

    def test_subtitles_full_comedy_remix_preset(self):
        """Verifies subtitle generation under full_comedy_remix preset applies safe margins and styles."""
        words_data = [
            {"word": "EAT", "start": 0.5, "end": 1.0},
            {"word": "EVERYTHING", "start": 1.0, "end": 1.8}
        ]
        out_ass = config.TEMP_DIR / "test_comedy_remix.ass"

        self.sub_gen.generate_ass(
            words_data=words_data,
            clip_start=0.0,
            clip_end=3.0,
            output_file=out_ass,
            preset="full_comedy_remix"
        )

        self.assertTrue(out_ass.exists())
        content = out_ass.read_text(encoding="utf-8")
        # Check Impact font and safe bottom margin 240
        self.assertIn("Impact", content)
        self.assertIn(",240,1", content)
        out_ass.unlink(missing_ok=True)

    def test_halftone_texture_asset(self):
        """Verifies halftone dot texture generator produces a valid transparent MOV."""
        ht_path = config.TEMP_DIR / "test_ht.mov"
        self.anim_engine.generate_halftone_texture(
            output_path=ht_path,
            duration=0.5,
            fps=15
        )
        self.assertTrue(ht_path.exists())
        self.assertGreater(ht_path.stat().st_size, 1000)
    def test_animated_emoji_engine_planning_and_mov_rendering(self):
        """Verifies that AnimatedEmojiEngine extracts assets, plans contextual reactions, and renders 32-bit transparent MOVs."""
        from src.emoji_anim import AnimatedEmojiEngine
        engine = AnimatedEmojiEngine()
        engine.ensure_emoji_assets()

        # Check asset creation
        surprise_asset = engine.asset_dir / "surprise_astonished.png"
        self.assertTrue(surprise_asset.exists(), "Surprise emoji PNG asset missing")

        # Plan reactions
        words_data = [
            {"word": "bought", "start": 0.5, "end": 0.8},
            {"word": "million", "start": 6.8, "end": 7.3},
            {"word": "dollars", "start": 7.3, "end": 7.8},
            {"word": "everything", "start": 11.5, "end": 12.0},
            {"word": "calories", "start": 18.0, "end": 18.8},
            {"word": "impossible", "start": 21.0, "end": 22.0},
        ]
        reactions = engine.plan_emoji_reactions(words_data, clip_start=0.0, clip_end=23.85)
        self.assertGreaterEqual(len(reactions), 3, "Should plan at least 3 contextual emojis")

        # Render a test animated emoji MOV
        test_mov = config.TEMP_DIR / "test_emoji_anim.mov"
        rxn = reactions[0]
        engine.render_animated_emoji_mov(
            reaction=rxn,
            output_mov=test_mov,
            fps=24
        )
        self.assertTrue(test_mov.exists(), "Animated emoji MOV was not created")
        self.assertGreater(test_mov.stat().st_size, 1000, "Animated emoji MOV file size is too small")
        test_mov.unlink(missing_ok=True)

    def test_clean_spoken_captions_no_tofu_boxes(self):
        """Verifies that ASS subtitle generation produces clean spoken dialogue without raw Unicode emojis."""
        words_data = [
            {"word": "I", "start": 0.0, "end": 0.3},
            {"word": "BOUGHT", "start": 0.3, "end": 0.6},
            {"word": "THIS", "start": 0.6, "end": 0.9},
            {"word": "STORE", "start": 0.9, "end": 1.2},
        ]
        out_ass = config.TEMP_DIR / "test_clean_subs.ass"
        self.sub_gen.generate_ass(
            words_data=words_data,
            clip_start=0.0,
            clip_end=2.0,
            output_file=out_ass,
            preset="full_comedy_remix"
        )
        self.assertTrue(out_ass.exists())
        content = out_ass.read_text(encoding="utf-8")

        # Check dialogue events contain ONLY words, no raw emoji chars like 🛒, 💰, etc.
        emojis_to_check = ["🛒", "💰", "🤔", "🔥", "🤯", "😲", "🎉", "😳", "😬", "😂", "🤨", "💥"]
        for em in emojis_to_check:
            self.assertNotIn(em, content, f"Raw emoji {em} found in ASS subtitle output!")
        out_ass.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
