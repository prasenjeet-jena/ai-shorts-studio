import unittest
from pathlib import Path
import json

from src.sound import SoundEngine
from src.storyboard import StoryRemixEngine, StoryRemixStoryboard, FreezeFrameBeat
from src.effects import VisualEffectEngine, EFFECT_REGISTRY


class TestStoryRemix(unittest.TestCase):

    def setUp(self):
        self.sound_engine = SoundEngine()
        self.story_engine = StoryRemixEngine()
        self.effect_engine = VisualEffectEngine()

        # Realistic words representing Candidate 1 of MrBeast video
        self.sample_words = [
            {"word": "I", "start": 0.0, "end": 0.3},
            {"word": "just", "start": 0.3, "end": 0.48},
            {"word": "bought", "start": 0.48, "end": 0.68},
            {"word": "this", "start": 0.68, "end": 0.92},
            {"word": "entire", "start": 0.92, "end": 1.24},
            {"word": "grocery", "start": 1.24, "end": 1.6},
            {"word": "store,", "start": 1.6, "end": 2.04},
            {"word": "and", "start": 2.2, "end": 2.3},
            {"word": "no", "start": 2.3, "end": 2.54},
            {"word": "joke,", "start": 2.54, "end": 2.7},
            {"word": "this", "start": 2.94, "end": 3.02},
            {"word": "is", "start": 3.02, "end": 3.12},
            {"word": "crazy.", "start": 3.12, "end": 3.34},
            {"word": "But", "start": 3.88, "end": 3.88},
            {"word": "if", "start": 3.88, "end": 3.98},
            {"word": "these", "start": 3.98, "end": 4.24},
            {"word": "four", "start": 4.24, "end": 4.5},
            {"word": "contestants", "start": 4.5, "end": 4.82},
            {"word": "eat", "start": 4.82, "end": 5.2},
            {"word": "everything", "start": 5.2, "end": 5.6},
            {"word": "inside,", "start": 5.6, "end": 6.08},
            {"word": "they", "start": 6.36, "end": 6.54},
            {"word": "win", "start": 6.54, "end": 6.7},
            {"word": "a", "start": 6.7, "end": 6.84},
            {"word": "million", "start": 6.84, "end": 7.2},
            {"word": "dollars.", "start": 7.2, "end": 7.68},
            {"word": "Oh", "start": 7.9, "end": 7.92},
            {"word": "my", "start": 7.92, "end": 8.58},
            {"word": "god!", "start": 8.58, "end": 8.9},
            {"word": "You", "start": 9.9, "end": 10.08},
            {"word": "mean", "start": 10.08, "end": 10.22},
            {"word": "like", "start": 10.22, "end": 10.4},
            {"word": "everything?", "start": 10.4, "end": 10.72},
            {"word": "Everything.", "start": 11.44, "end": 11.92},
            {"word": "Millions", "start": 17.7, "end": 17.74},
            {"word": "and", "start": 17.74, "end": 18.34},
            {"word": "millions", "start": 18.34, "end": 18.74},
            {"word": "of", "start": 18.74, "end": 19.12},
            {"word": "calories", "start": 19.12, "end": 19.36},
            {"word": "That's", "start": 21.92, "end": 22.4},
            {"word": "impossible.", "start": 22.4, "end": 22.62},
            {"word": "Wow.", "start": 23.46, "end": 23.66}
        ]

    def test_sound_engine_synthesis_and_files(self):
        """Verifies procedural sound effect generation and file validity."""
        cues = ["record_scratch", "comic_pop", "dramatic_hit", "comedic_sting", "whoosh"]
        for c in cues:
            p = self.sound_engine.get_cue_path(c)
            self.assertTrue(p.exists(), f"Missing audio cue: {c}")
            self.assertGreater(p.stat().st_size, 5000, f"Audio cue {c} is too small")

    def test_storyboard_creation_and_freeze_beat(self):
        """Verifies that StoryRemixEngine detects the comedic pause at 8.9s."""
        sb = self.story_engine.create_storyboard(
            words_data=self.sample_words,
            clip_start=0.0,
            clip_end=23.85,
            clip_spec={"id": "c1_test", "hook": "Grocery Store Survival"}
        )
        self.assertIsNotNone(sb.freeze_beat)
        fb = sb.freeze_beat
        self.assertEqual(fb.start_time, 8.9)
        self.assertEqual(fb.duration, 1.0)
        self.assertEqual(fb.sound_cue, "record_scratch")
        self.assertEqual(fb.visual_treatment, "comic_spotlight")
        self.assertIn("canned tuna", fb.editorial_commentary)
        self.assertEqual(fb.presentation_style, "thought_bubble")

    def test_shift_words_for_freeze(self):
        """Verifies downstream timestamp shifting preserves caption synchronization."""
        shifted = self.story_engine.shift_words_for_freeze(
            words_data=self.sample_words,
            freeze_start=8.9,
            freeze_duration=1.0
        )
        # Word at 7.2s should remain unchanged
        w_before = next(w for w in shifted if w["word"] == "dollars.")
        self.assertEqual(w_before["start"], 7.2)
        self.assertEqual(w_before["end"], 7.68)

        # Word originally at 9.9s should now be shifted to 10.9s
        w_after = next(w for w in shifted if w["word"] == "You")
        self.assertEqual(w_after["start"], 10.9)
        self.assertEqual(w_after["end"], 11.08)

    def test_cartoon_accents_and_sound_cues(self):
        """Verifies cartoon accents and timeline sound cues."""
        sb = self.story_engine.create_storyboard(
            words_data=self.sample_words,
            clip_start=0.0,
            clip_end=23.85
        )
        self.assertGreaterEqual(len(sb.cartoon_accents), 2)
        accent_symbols = [a.symbol for a in sb.cartoon_accents]
        self.assertIn("(!)", accent_symbols)  # Sweat drop / panic indicator
        self.assertIn(">>", accent_symbols)   # Calorie counter indicator

        # Check sound cues timeline
        cues = [c["cue"] for c in sb.sound_cues]
        self.assertIn("record_scratch", cues)
        self.assertIn("comic_pop", cues)

    def test_originality_assessment(self):
        """Verifies originality score and transformational differentiators."""
        sb = self.story_engine.create_storyboard(
            words_data=self.sample_words,
            clip_start=0.0,
            clip_end=23.85
        )
        orig = sb.originality
        self.assertGreaterEqual(orig.overall_score, 0.85)
        self.assertGreaterEqual(len(orig.differentiators), 4)
        self.assertIn("freeze", orig.editorial_rationale.lower())

    def test_storyboard_ass_dialogue_compilation(self):
        """Verifies compilation of Storyboard into Layer 2 ASS Dialogues."""
        sb = self.story_engine.create_storyboard(
            words_data=self.sample_words,
            clip_start=0.0,
            clip_end=23.85
        )
        lines = self.effect_engine.compile_storyboard_ass_events(sb, clip_duration=24.85)
        self.assertGreaterEqual(len(lines), 3)

        # Check for freeze thought bubble
        bubble_lines = [l for l in lines if "RemixBubble" in l]
        self.assertEqual(len(bubble_lines), 1)
        self.assertIn("THOUGHT:", bubble_lines[0])
        self.assertIn("canned tuna", bubble_lines[0])

        # Check for cartoon accent
        aside_lines = [l for l in lines if "RemixAside" in l]
        self.assertGreaterEqual(len(aside_lines), 2)

    def test_video_animation_beats_planning(self):
        """Verifies coordinated planning of real video animations in storyboard."""
        sb = self.story_engine.create_storyboard(
            words_data=self.sample_words,
            clip_start=0.0,
            clip_end=23.85
        )
        self.assertGreaterEqual(len(sb.video_animations), 4)
        anim_types = [a.anim_type for a in sb.video_animations]
        self.assertIn("radial_burst", anim_types)
        self.assertIn("screen_shake", anim_types)
        self.assertIn("speed_lines", anim_types)
        self.assertIn("cel_shaded_video", anim_types)

    def test_animation_engine_assets_and_comic_transform(self):
        """Verifies procedural asset generation and OpenCV comic frame transformation."""
        from src.anim import AnimationEngine
        import numpy as np

        engine = AnimationEngine()
        self.assertTrue(engine.radial_burst_path.exists())
        self.assertTrue(engine.speed_lines_path.exists())

        # Test comic transformation on synthetic frame
        dummy = np.full((1920, 1080, 3), 128, dtype=np.uint8)
        comic = engine.transform_frame_to_comic(dummy, add_border=True)
        self.assertEqual(comic.shape, (1920, 1080, 3))


if __name__ == "__main__":
    unittest.main()
