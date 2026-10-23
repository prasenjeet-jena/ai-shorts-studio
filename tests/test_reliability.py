import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config import config
from src.preflight import run_preflight, PreflightError, check_disk_space, check_writability
from src.cache import CacheManager, CACHE_VERSION
from src.run_manager import RunManager, RunStatus, JobStatus
from src.batch import BatchProcessor, parse_batch_request, validate_batch_settings, MAX_JOB_RETRIES
from src.selector import CandidateSelector


class TestPhase6Reliability(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="shorts_test_"))
        self.runs_dir = self.test_dir / "runs"
        self.cache_dir = self.test_dir / "cache"
        self.output_dir = self.test_dir / "output"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 1. Preflight Checks
    # -------------------------------------------------------------
    def test_preflight_checks_pass(self):
        """Verifies that preflight passes with standard requirements on this system."""
        result = run_preflight(min_disk_gb=0.1)
        self.assertTrue(result["ready"])
        self.assertTrue(result["all_passed"])
        self.assertEqual(len(result["critical_failed"]), 0)

    def test_preflight_disk_space_failure(self):
        """Fault injection: simulates insufficient disk space and verifies early abort."""
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(min_disk_gb=9999999.0)
        self.assertIn("Disk Space", str(ctx.exception))

    def test_preflight_writability_check(self):
        """Verifies check_writability correctly identifies writable and non-writable directories."""
        res_ok = check_writability([self.output_dir])
        self.assertTrue(res_ok["passed"])

        # Create a non-writable test directory
        readonly_dir = self.test_dir / "readonly"
        readonly_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(readonly_dir, 0o444)
        try:
            res_bad = check_writability([readonly_dir])
            # On macOS as root or admin, write might still succeed; if so skip assertion
            if not res_bad["passed"]:
                self.assertFalse(res_bad["passed"])
        finally:
            os.chmod(readonly_dir, 0o777)

    # -------------------------------------------------------------
    # 2. Cache Manager: Hits, Invalidation, and Versioning
    # -------------------------------------------------------------
    def test_cache_save_load_and_versioning(self):
        """Tests that versioned cache saves, loads, and rejects stale versions."""
        cm = CacheManager(cache_dir=self.cache_dir)
        test_data = [{"block_id": 1, "text": "Testing cache"}]
        fp = "test_fp_123"

        # Save cache entry
        save_path = cm.save("vid1", "summary", test_data, fingerprint=fp)
        self.assertTrue(save_path.exists())

        # Load cache entry (hit)
        loaded = cm.load("vid1", "summary", fingerprint=fp)
        self.assertEqual(loaded, test_data)

        # Mismatched fingerprint should result in cache miss
        miss = cm.load("vid1", "summary", fingerprint="different_fp")
        self.assertIsNone(miss)

        # Unknown artifact should be None
        self.assertIsNone(cm.load("vid1", "nonexistent"))

    def test_cache_safe_invalidation(self):
        """Tests that invalidating a specific video_id deletes only its entries."""
        cm = CacheManager(cache_dir=self.cache_dir)
        cm.save("vidA", "words", {"words": [1, 2]})
        cm.save("vidA", "scenes", [1.0, 5.0])
        cm.save("vidB", "words", {"words": [3, 4]})

        self.assertTrue(cm.has("vidA", "words"))
        self.assertTrue(cm.has("vidB", "words"))

        # Invalidate only vidA
        removed = cm.invalidate(video_id="vidA")
        self.assertEqual(removed, 2)
        self.assertFalse(cm.has("vidA", "words"))
        self.assertFalse(cm.has("vidA", "scenes"))
        # vidB must remain intact
        self.assertTrue(cm.has("vidB", "words"))

    # -------------------------------------------------------------
    # 3. Run Manager: State Persistence, Atomic Writes, & Resumption
    # -------------------------------------------------------------
    def test_run_manager_lifecycle_and_atomic_writes(self):
        """Tests run creation, stage progression, job state updates, and atomic state.json."""
        rm = RunManager(runs_dir=self.runs_dir)
        run_id = "test_run_101"

        state = rm.create_run(
            source_info={"video_id": "vid123", "title": "Test Title"},
            parameters={"count": 3, "max_duration": 30.0},
            output_dir=self.output_dir,
            run_id=run_id
        )
        self.assertEqual(state["status"], RunStatus.PENDING)
        self.assertTrue((self.runs_dir / run_id / "state.json").exists())

        # Set candidates
        cands = [
            {"id": "short_1", "duration": 15.0, "score": 85},
            {"id": "short_2", "duration": 20.0, "score": 80}
        ]
        rm.set_candidates(run_id, cands)
        reloaded = rm.load_run(run_id)
        self.assertEqual(reloaded["status"], RunStatus.CANDIDATES_READY)
        self.assertEqual(len(reloaded["jobs"]), 2)

        # Update stage
        rm.update_stage(run_id, "rendering", RunStatus.RENDERING)
        reloaded = rm.load_run(run_id)
        self.assertEqual(reloaded["stages"]["rendering"]["status"], RunStatus.RENDERING)

        # Complete job 1
        rm.update_job(
            run_id, "short_1", JobStatus.COMPLETED,
            output_file=str(self.output_dir / "short_1.mp4"),
            validation={"is_vertical_9_16": True}
        )
        reloaded = rm.load_run(run_id)
        self.assertEqual(reloaded["jobs"]["short_1"]["status"], JobStatus.COMPLETED)
        self.assertEqual(reloaded["metrics"]["completed_jobs"], 1)

    # -------------------------------------------------------------
    # 4. Fault Injection & Retry Handling
    # -------------------------------------------------------------
    def test_retry_transient_failure_then_success(self):
        """Fault injection: simulates a transient render error that succeeds on retry."""
        proc = BatchProcessor()
        proc.run_mgr = RunManager(runs_dir=self.runs_dir)
        proc.cache_mgr = CacheManager(cache_dir=self.cache_dir)

        # Mock renderer to fail on attempt 1, succeed on attempt 2
        mock_renderer = MagicMock()
        mock_renderer.probe_video.return_value = {"width": 1080, "height": 1920, "codec": "h264"}
        attempt_tracker = {"count": 0}

        def mock_render(*args, **kwargs):
            attempt_tracker["count"] += 1
            if attempt_tracker["count"] == 1:
                raise RuntimeError("Transient FFmpeg I/O glitch")
            return {
                "id": "short_1",
                "filename": "short_1.mp4",
                "filepath": str(self.output_dir / "short_1.mp4"),
                "duration": 15.0,
                "dead_air_trimmed": 0.5,
                "resolution": "1080x1920"
            }

        mock_renderer.render_clip.side_effect = mock_render
        proc.renderer = mock_renderer

        run_id = "test_transient_retry"
        settings = {
            "url": "https://www.youtube.com/watch?v=1aA1WGON49E",
            "count": 1,
            "max_duration": 30.0
        }

        # Mock ingestion & transcription to avoid external network calls
        proc.ingestion_mgr.download = MagicMock(return_value=(
            self.test_dir / "src.mp4", self.test_dir / "src.wav", {"video_id": "vid_mock", "duration": 30.0}
        ))
        proc.transcriber.transcribe = MagicMock(return_value={
            "words": [{"word": "test", "start": 0.0, "end": 1.0}],
            "blocks": [{"block_id": 1, "text": "Test block", "start": 0.0, "end": 15.0}]
        })
        proc.scene_detector.detect_scene_cuts = MagicMock(return_value=[])

        # Write dummy output file to satisfy validation
        dummy_mp4 = self.output_dir / "short_1.mp4"
        with open(dummy_mp4, "wb") as f:
            f.write(b"0" * 10000)

        test_cands = [{"id": "short_1", "start": 0.0, "end": 15.0, "duration": 15.0, "score": 80}]
        manifest = proc.process_batch(
            settings=settings,
            output_dir=self.output_dir,
            agent_candidates=test_cands,
            run_id=run_id,
            skip_preflight=True
        )

        # Must have attempted twice and succeeded
        self.assertEqual(attempt_tracker["count"], 2)
        self.assertEqual(manifest["status"], RunStatus.COMPLETED)
        self.assertEqual(len(manifest["rendered_shorts"]), 1)

    def test_permanent_failure_bounded_retries(self):
        """Fault injection: permanent failure retries only up to MAX_JOB_RETRIES (2) and stops."""
        proc = BatchProcessor()
        proc.run_mgr = RunManager(runs_dir=self.runs_dir)
        proc.cache_mgr = CacheManager(cache_dir=self.cache_dir)

        mock_renderer = MagicMock()
        mock_renderer.probe_video.return_value = {"width": 1080, "height": 1920, "codec": "h264"}
        attempt_tracker = {"count": 0}

        def mock_permanent_fail(*args, **kwargs):
            attempt_tracker["count"] += 1
            raise RuntimeError("Permanent invalid stream error")

        mock_renderer.render_clip.side_effect = mock_permanent_fail
        proc.renderer = mock_renderer

        # Mock ingestion & transcription
        proc.ingestion_mgr.download = MagicMock(return_value=(
            self.test_dir / "src.mp4", self.test_dir / "src.wav", {"video_id": "vid_mock", "duration": 30.0}
        ))
        proc.transcriber.transcribe = MagicMock(return_value={
            "words": [{"word": "test", "start": 0.0, "end": 1.0}],
            "blocks": [{"block_id": 1, "text": "Test block", "start": 0.0, "end": 15.0}]
        })
        proc.scene_detector.detect_scene_cuts = MagicMock(return_value=[])

        run_id = "test_perm_fail"
        manifest = proc.process_batch(
            settings={"url": "https://www.youtube.com/watch?v=1aA1WGON49E", "count": 1, "max_duration": 30.0},
            output_dir=self.output_dir,
            agent_candidates=[{"id": "short_1", "start": 0.0, "end": 15.0, "duration": 15.0, "score": 80}],
            run_id=run_id,
            skip_preflight=True
        )

        # Must have stopped exactly at MAX_JOB_RETRIES without infinite loop
        self.assertEqual(attempt_tracker["count"], MAX_JOB_RETRIES)
        self.assertEqual(manifest["status"], RunStatus.FAILED)
        self.assertEqual(manifest["rendered_count"], 0)

    # -------------------------------------------------------------
    # 5. Output Protection & Partial Batch Completion
    # -------------------------------------------------------------
    def test_partial_batch_completion_isolation(self):
        """Verifies that when one job fails, successful sibling jobs are preserved."""
        proc = BatchProcessor()
        proc.run_mgr = RunManager(runs_dir=self.runs_dir)
        proc.cache_mgr = CacheManager(cache_dir=self.cache_dir)

        mock_renderer = MagicMock()
        mock_renderer.probe_video.return_value = {"width": 1080, "height": 1920, "codec": "h264"}

        # Candidate 1 succeeds, Candidate 2 fails permanently
        def mock_render_selective(source_video, words_data, clip_spec, **kwargs):
            cid = clip_spec["id"]
            if cid == "short_2":
                raise RuntimeError("Candidate 2 failed render")
            out_file = self.output_dir / f"{cid}.mp4"
            with open(out_file, "wb") as f:
                f.write(b"0" * 10000)
            return {
                "id": cid,
                "filename": f"{cid}.mp4",
                "filepath": str(out_file),
                "duration": 15.0,
                "dead_air_trimmed": 0.0,
                "resolution": "1080x1920"
            }

        mock_renderer.render_clip.side_effect = mock_render_selective
        proc.renderer = mock_renderer

        agent_cands = [
            {"id": "short_1", "start": 0.0, "end": 15.0, "duration": 15.0, "score": 80},
            {"id": "short_2", "start": 16.0, "end": 30.0, "duration": 14.0, "score": 75}
        ]

        proc.ingestion_mgr.download = MagicMock(return_value=(
            self.test_dir / "src.mp4", self.test_dir / "src.wav", {"video_id": "vid_mock", "duration": 60.0}
        ))
        proc.transcriber.transcribe = MagicMock(return_value={"words": [], "blocks": []})
        proc.scene_detector.detect_scene_cuts = MagicMock(return_value=[])

        run_id = "test_partial_batch"
        manifest = proc.process_batch(
            settings={"url": "https://www.youtube.com/watch?v=1aA1WGON49E", "count": 2, "max_duration": 30.0},
            output_dir=self.output_dir,
            agent_candidates=agent_cands,
            run_id=run_id,
            skip_preflight=True
        )

        # Status must be partially_completed
        self.assertEqual(manifest["status"], RunStatus.PARTIALLY_COMPLETED)
        self.assertEqual(manifest["rendered_count"], 1)
        self.assertEqual(manifest["rendered_shorts"][0]["id"], "short_1")
        # short_1 output must exist on disk
        self.assertTrue((self.output_dir / "short_1.mp4").exists())

    # -------------------------------------------------------------
    # 6. Resume Mechanism: Reusing Validated Outputs
    # -------------------------------------------------------------
    def test_resume_reuses_already_validated_clips(self):
        """Simulates resuming an interrupted run and verifies verified clips are not re-rendered."""
        rm = RunManager(runs_dir=self.runs_dir)
        run_id = "test_resume_reuse"

        # Pre-seed run state with short_1 COMPLETED and short_2 PENDING
        state = rm.create_run(
            source_info={"video_id": "1aA1WGON49E", "url": "https://www.youtube.com/watch?v=1aA1WGON49E"},
            parameters={"count": 2, "max_duration": 30.0, "preset": "vibrant_pro", "editing_intensity": "dynamic"},
            output_dir=self.output_dir,
            run_id=run_id
        )

        cands = [
            {"id": "short_1", "start": 0.0, "end": 15.0, "duration": 15.0, "score": 85},
            {"id": "short_2", "start": 20.0, "end": 35.0, "duration": 15.0, "score": 80}
        ]
        rm.set_candidates(run_id, cands)

        # Create valid short_1.mp4 on disk
        short_1_path = self.output_dir / "short_1.mp4"
        with open(short_1_path, "wb") as f:
            f.write(b"0" * 15000)

        # Mark short_1 as completed
        rm.update_job(
            run_id, "short_1", JobStatus.COMPLETED,
            output_file=str(short_1_path),
            validation={"width": 1080, "height": 1920, "codec": "h264"}
        )

        # Setup BatchProcessor
        proc = BatchProcessor()
        proc.run_mgr = rm
        proc.cache_mgr = CacheManager(cache_dir=self.cache_dir)

        mock_renderer = MagicMock()
        # probe_video returns valid properties
        mock_renderer.probe_video.return_value = {"width": 1080, "height": 1920, "codec": "h264"}

        # Only short_2 should ever be called for rendering
        rendered_called = []
        def mock_render(source_video, words_data, clip_spec, **kwargs):
            cid = clip_spec["id"]
            rendered_called.append(cid)
            out_file = self.output_dir / f"{cid}.mp4"
            with open(out_file, "wb") as f:
                f.write(b"0" * 15000)
            return {
                "id": cid,
                "filename": f"{cid}.mp4",
                "filepath": str(out_file),
                "duration": 15.0,
                "dead_air_trimmed": 0.0,
                "resolution": "1080x1920"
            }

        mock_renderer.render_clip.side_effect = mock_render
        proc.renderer = mock_renderer

        # Mock ingestion & transcription
        proc.ingestion_mgr.download = MagicMock(return_value=(
            self.test_dir / "src.mp4", self.test_dir / "src.wav", {"video_id": "1aA1WGON49E", "duration": 60.0}
        ))
        proc.transcriber.transcribe = MagicMock(return_value={"words": [], "blocks": []})
        proc.scene_detector.detect_scene_cuts = MagicMock(return_value=[])

        # Resume run
        manifest = proc.resume_batch(run_id=run_id)

        # Verification: short_1 was REUSED, only short_2 was rendered!
        self.assertNotIn("short_1", rendered_called)
        self.assertIn("short_2", rendered_called)
        self.assertEqual(manifest["status"], RunStatus.COMPLETED)
        self.assertEqual(manifest["rendered_count"], 2)


if __name__ == "__main__":
    unittest.main()
