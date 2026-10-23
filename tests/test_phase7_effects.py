import unittest
from pathlib import Path
from src.events import VisualEventDetector, VisualEvent
from src.effects import VisualEffectEngine, EFFECT_REGISTRY, format_ass_time


class TestPhase7Effects(unittest.TestCase):

    def setUp(self):
        self.detector = VisualEventDetector()
        self.engine = VisualEffectEngine()

        # Sample words matching Candidate 1 of MrBeast video
        self.sample_words = [
            {"word": "I", "start": 0.0, "end": 0.3},
            {"word": "just", "start": 0.3, "end": 0.5},
            {"word": "bought", "start": 0.5, "end": 0.7},
            {"word": "this", "start": 0.7, "end": 0.9},
            {"word": "entire", "start": 0.9, "end": 1.2},
            {"word": "grocery", "start": 1.2, "end": 1.6},
            {"word": "store,", "start": 1.6, "end": 2.0},
            {"word": "if", "start": 3.9, "end": 4.0},
            {"word": "these", "start": 4.0, "end": 4.2},
            {"word": "four", "start": 4.2, "end": 4.5},
            {"word": "contestants", "start": 4.5, "end": 4.8},
            {"word": "eat", "start": 4.8, "end": 5.2},
            {"word": "everything", "start": 5.2, "end": 5.6},
            {"word": "inside,", "start": 5.6, "end": 6.1},
            {"word": "they", "start": 6.4, "end": 6.5},
            {"word": "win", "start": 6.5, "end": 6.7},
            {"word": "a", "start": 6.7, "end": 6.8},
            {"word": "million", "start": 6.8, "end": 7.2},
            {"word": "dollars.", "start": 7.2, "end": 7.7},
            {"word": "Oh", "start": 7.9, "end": 8.0},
            {"word": "my", "start": 8.0, "end": 8.5},
            {"word": "god!", "start": 8.5, "end": 8.9},
            {"word": "You", "start": 9.9, "end": 10.1},
            {"word": "mean", "start": 10.1, "end": 10.2},
            {"word": "like", "start": 10.2, "end": 10.4},
            {"word": "everything?", "start": 10.4, "end": 10.7},
            {"word": "Everything.", "start": 11.4, "end": 11.9},
            {"word": "Millions", "start": 17.7, "end": 18.0},
            {"word": "of", "start": 18.0, "end": 18.3},
            {"word": "calories", "start": 18.3, "end": 19.3},
            {"word": "That's", "start": 21.9, "end": 22.4},
            {"word": "impossible.", "start": 22.4, "end": 22.7},
            {"word": "Wow.", "start": 23.4, "end": 23.6}
        ]

    def test_detect_money_reward_event(self):
        """Verifies detection and grounding of monetary prize amounts."""
        events = self.detector.detect_events(
            words_data=self.sample_words,
            clip_start=0.0,
            clip_end=24.0,
            intensity="dynamic"
        )
        money_evts = [e for e in events if e.event_type == "money_reward"]
        self.assertGreaterEqual(len(money_evts), 1)
        m = money_evts[0]
        self.assertEqual(m.recommended_effect, "money_reward_badge")
        self.assertIn("$1,000,000", m.parameters.get("badge_text", ""))
        self.assertGreaterEqual(m.confidence, 0.90)
        self.assertTrue(5.0 <= m.start <= 8.5)

    def test_detect_intense_reaction_event(self):
        """Verifies detection of high-energy exclamation beats."""
        events = self.detector.detect_events(
            words_data=self.sample_words,
            clip_start=0.0,
            clip_end=24.0,
            intensity="dynamic"
        )
        intense_evts = [e for e in events if e.event_type == "intense_reaction"]
        self.assertGreaterEqual(len(intense_evts), 1)
        ievt = intense_evts[0]
        self.assertIn("impact", ievt.recommended_effect)
        self.assertTrue(7.5 <= ievt.start <= 9.0)

    def test_detect_surprise_reveal_event(self):
        """Verifies detection of curiosity reveal and surprise question beats."""
        events = self.detector.detect_events(
            words_data=self.sample_words,
            clip_start=0.0,
            clip_end=24.0,
            intensity="dynamic"
        )
        surp_evts = [e for e in events if e.event_type == "surprise_reveal"]
        self.assertGreaterEqual(len(surp_evts), 1)
        sevt = surp_evts[0]
        self.assertEqual(sevt.recommended_effect, "surprise_burst")
        self.assertIn("?", sevt.parameters.get("badge_text", ""))
        self.assertTrue(9.5 <= sevt.start <= 11.5)

    def test_conflict_resolution_and_pacing(self):
        """Verifies that events are spaced and capped according to intensity preset."""
        # Dynamic intensity allows up to 4 events
        evts_dyn = self.detector.detect_events(
            words_data=self.sample_words,
            clip_start=0.0,
            clip_end=24.0,
            intensity="dynamic"
        )
        self.assertLessEqual(len(evts_dyn), 4)

        # Subtle intensity allows at most 1 event
        evts_sub = self.detector.detect_events(
            words_data=self.sample_words,
            clip_start=0.0,
            clip_end=24.0,
            intensity="subtle"
        )
        self.assertLessEqual(len(evts_sub), 1)

    def test_effect_engine_ass_styles_and_dialogues(self):
        """Verifies generation of ASS Style lines and Layer 2 Dialogue events."""
        evts = self.detector.detect_events(
            words_data=self.sample_words,
            clip_start=0.0,
            clip_end=24.0,
            intensity="dynamic"
        )
        styles = self.engine.get_ass_styles(preset="vibrant_pro")
        self.assertTrue(any("EvtMoney" in s for s in styles))
        self.assertTrue(any("EvtSurprise" in s for s in styles))

        dialogues = self.engine.compile_ass_events(events=evts, clip_start=0.0, preset="vibrant_pro")
        self.assertGreaterEqual(len(dialogues), 2)
        # All event dialogues must be on Layer 2 (above base captions)
        for d in dialogues:
            self.assertTrue(d.startswith("Dialogue: 2,"))
            self.assertIn("\\an8", d)  # Anchored in safe top header zone
            self.assertIn("\\fscx", d) # Contains pop-in animation

    def test_video_filters_compilation(self):
        """Verifies generation of FFmpeg video filter expressions with escaped commas."""
        confrontation_evt = VisualEvent(
            event_id="evt_test",
            event_type="intense_reaction",
            start=2.0,
            end=3.2,
            duration=1.2,
            confidence=0.92,
            evidence="Test panic",
            recommended_effect="impact_accent",
            intensity="high",
            transcript_excerpt="Oh no!",
            parameters={"red_edge_flash": True}
        )
        filters = self.engine.compile_video_filters(events=[confrontation_evt], preset="vibrant_pro")
        self.assertEqual(len(filters), 1)
        f_expr = filters[0]
        self.assertIn("drawbox=", f_expr)
        self.assertIn("between(t\\,2.00\\,", f_expr) # Commas escaped for FFmpeg filtergraph

        # Clean Pro disables video flashes
        filters_clean = self.engine.compile_video_filters(events=[confrontation_evt], preset="clean_pro")
        self.assertEqual(len(filters_clean), 0)


if __name__ == "__main__":
    unittest.main()
