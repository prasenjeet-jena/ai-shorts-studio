import unittest
import importlib

class TestSmokeImports(unittest.TestCase):
    """
    Lightweight smoke test verifying that all core modules import cleanly
    without side effects, network requests, or model downloads.
    """

    def test_imports(self):
        modules = [
            "src.config",
            "src.logger",
            "src.preflight",
            "src.cache",
            "src.ingestion",
            "src.transcription",
            "src.scene",
            "src.selector",
            "src.events",
            "src.framing",
            "src.pacing",
            "src.subtitles",
            "src.sound",
            "src.anim",
            "src.reactions",
            "src.emoji_anim",
            "src.storyboard",
            "src.comedy_timeline",
            "src.cartoon_engine",
            "src.rendering",
            "src.run_manager",
            "src.batch",
            "src.cli",
            "generate",
        ]
        for mod in modules:
            with self.subTest(module=mod):
                m = importlib.import_module(mod)
                self.assertIsNotNone(m)

if __name__ == "__main__":
    unittest.main()
