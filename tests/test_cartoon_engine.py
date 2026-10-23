import unittest
import numpy as np
from pathlib import Path

from src.cartoon_engine import AnimeGANv2CartoonEngine
from src.subtitles import SubtitleGenerator
from src.config import config


class TestPhase10CartoonEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AnimeGANv2CartoonEngine()
        self.sub_gen = SubtitleGenerator()

    def test_model_presence_and_session(self):
        """Verifies that the AnimeGANv2 Hayao model is present and loads into an ONNX session."""
        expected_path = self.engine.model_dir / self.engine.MODELS["hayao"]["filename"]
        if not expected_path.exists():
            self.skipTest("AnimeGANv2 Hayao ONNX model not pre-downloaded; skipping download in unit test")
        model_path = self.engine.ensure_model()
        self.assertTrue(model_path.exists())
        self.assertGreater(model_path.stat().st_size, 5_000_000)

        session, input_name = self.engine._get_session()
        self.assertIsNotNone(session)
        inputs = session.get_inputs()
        self.assertGreaterEqual(len(inputs), 1)
        self.assertEqual(inputs[0].name, input_name)

    def test_classic_toon_presence_and_frame(self):
        """Verifies Tom & Jerry / classic_toon style model loading and frame cartoonization."""
        expected_path = self.engine.model_dir / self.engine.MODELS["classic_toon"]["filename"]
        if not expected_path.exists():
            self.skipTest("Whitebox cartoonizer ONNX model not pre-downloaded; skipping download in unit test")
        model_path = self.engine.ensure_model("classic_toon")
        self.assertTrue(model_path.exists())
        self.assertGreater(model_path.stat().st_size, 5_000_000)

        session, input_name = self.engine._get_session("classic_toon")
        self.assertIsNotNone(session)

        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        test_frame[:, :] = (120, 180, 220)
        test_frame[300:700, 800:1200] = (200, 50, 50)

        out_frame = self.engine.process_frame(test_frame, style="tom_and_jerry")
        self.assertIsNotNone(out_frame)
        self.assertEqual(out_frame.shape, (1920, 1080, 3))
        self.assertEqual(out_frame.dtype, np.uint8)

    def test_process_single_frame(self):
        """Verifies process_frame takes arbitrary BGR frame and returns 1080x1920 cartoonized frame."""
        # Create a synthetic 1080x1920 test frame with patterns
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        test_frame[:, :] = (120, 180, 220)  # Soft skin/sky tone
        test_frame[300:700, 800:1200] = (200, 50, 50)  # Red box (shirt/object)
        test_frame[100:250, 900:1100] = (50, 200, 50)  # Green box (face/hair)

        out_frame = self.engine.process_frame(test_frame)
        self.assertIsNotNone(out_frame)
        self.assertEqual(out_frame.shape, (1920, 1080, 3))
        self.assertEqual(out_frame.dtype, np.uint8)
        # Verify cartoonized frame has rich color variation
        self.assertGreater(float(np.std(out_frame)), 5.0)

    def test_subtitles_full_cartoon_preset(self):
        """Verifies subtitle generation under full_cartoon preset applies bold comic margins and styles."""
        words_data = [
            {"word": "I", "start": 0.0, "end": 0.3},
            {"word": "BOUGHT", "start": 0.3, "end": 0.6},
            {"word": "THIS", "start": 0.6, "end": 0.9},
            {"word": "STORE", "start": 0.9, "end": 1.2},
        ]
        out_ass = config.TEMP_DIR / "test_cartoon_subs.ass"

        self.sub_gen.generate_ass(
            words_data=words_data,
            clip_start=0.0,
            clip_end=2.0,
            output_file=out_ass,
            preset="full_cartoon"
        )

        self.assertTrue(out_ass.exists())
        content = out_ass.read_text(encoding="utf-8")
        self.assertIn("Impact", content)
        self.assertIn(",240,1", content)
        out_ass.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
