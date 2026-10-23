import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config import config
from src.logger import PipelineLogger, default_logger

RUNS_DIR = config.STORAGE_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


class RunStatus:
    PENDING = "pending"
    ANALYZING = "analyzing"
    CANDIDATES_READY = "candidates_ready"
    RENDERING = "rendering"
    VALIDATING = "validating"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"


class JobStatus:
    PENDING = "pending"
    RENDERING = "rendering"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class RunManager:
    """
    Manages persistent state for batch and single runs.
    Enables reliable resumption, stage tracking, atomic metadata persistence,
    and output validation.
    """

    def __init__(self, runs_dir: Optional[Path] = None):
        self.runs_dir = Path(runs_dir) if runs_dir else RUNS_DIR
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def generate_run_id(self, video_id: Optional[str] = None) -> str:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        v_tag = f"_{video_id}" if video_id else ""
        return f"run_{timestamp_str}{v_tag}"

    def get_run_dir(self, run_id: str) -> Path:
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def get_state_file(self, run_id: str) -> Path:
        return self.get_run_dir(run_id) / "state.json"

    def create_run(
        self,
        source_info: Dict[str, Any],
        parameters: Dict[str, Any],
        output_dir: Optional[Path] = None,
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        vid = source_info.get("video_id") or "unknown"
        r_id = run_id or self.generate_run_id(vid)
        run_dir = self.get_run_dir(r_id)

        target_output = Path(output_dir) if output_dir else (config.BASE_DIR / "output" / "clips_batch")
        target_output.mkdir(parents=True, exist_ok=True)

        now = time.time()
        state = {
            "run_id": r_id,
            "created_at": now,
            "updated_at": now,
            "status": RunStatus.PENDING,
            "source": source_info,
            "parameters": parameters,
            "output_dir": str(target_output.resolve()),
            "manifest_path": str((target_output / "batch_manifest.json").resolve()),
            "stages": {
                "preflight": {"status": RunStatus.PENDING, "started_at": None, "ended_at": None, "error": None},
                "ingestion": {"status": RunStatus.PENDING, "started_at": None, "ended_at": None, "error": None},
                "transcription": {"status": RunStatus.PENDING, "started_at": None, "ended_at": None, "error": None},
                "scene_detection": {"status": RunStatus.PENDING, "started_at": None, "ended_at": None, "error": None},
                "candidate_selection": {"status": RunStatus.PENDING, "started_at": None, "ended_at": None, "error": None},
                "rendering": {"status": RunStatus.PENDING, "started_at": None, "ended_at": None, "error": None},
                "validation": {"status": RunStatus.PENDING, "started_at": None, "ended_at": None, "error": None}
            },
            "candidates": [],
            "jobs": {},
            "metrics": {
                "total_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "dead_air_trimmed_total_s": 0.0,
                "wall_time_s": 0.0
            }
        }

        self.save_run(state)
        default_logger.info(f"Initialized new run '{r_id}' with status {RunStatus.PENDING}.")
        return state

    def save_run(self, state: Dict[str, Any]):
        run_id = state["run_id"]
        state_file = self.get_state_file(run_id)
        state["updated_at"] = time.time()

        # Atomic file write
        tmp_file = state_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        tmp_file.replace(state_file)

    def load_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        state_file = self.get_state_file(run_id)
        if not state_file.exists():
            return None
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            default_logger.error(f"Error loading run state for '{run_id}': {e}")
            return None

    def update_stage(
        self,
        run_id: str,
        stage: str,
        status: str,
        error: Optional[str] = None
    ):
        state = self.load_run(run_id)
        if not state:
            return

        if stage not in state["stages"]:
            state["stages"][stage] = {}

        now = time.time()
        st = state["stages"][stage]
        st["status"] = status
        if status in [RunStatus.ANALYZING, RunStatus.RENDERING, "running"] and not st.get("started_at"):
            st["started_at"] = now
        if status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.PARTIALLY_COMPLETED]:
            st["ended_at"] = now
        if error:
            st["error"] = str(error)

        # Update high-level run status
        if status == RunStatus.FAILED and stage in ["preflight", "ingestion", "transcription"]:
            state["status"] = RunStatus.FAILED
        elif status in [RunStatus.ANALYZING, RunStatus.RENDERING, RunStatus.CANDIDATES_READY]:
            state["status"] = status

        self.save_run(state)
        default_logger.stage(stage, status, error or "")

    def set_candidates(self, run_id: str, candidates: List[Dict[str, Any]]):
        state = self.load_run(run_id)
        if not state:
            return

        state["candidates"] = candidates
        state["metrics"]["total_jobs"] = len(candidates)

        # Initialize job slots for each candidate
        jobs = {}
        for c in candidates:
            cid = c.get("id")
            existing = state.get("jobs", {}).get(cid)
            if existing:
                jobs[cid] = existing
            else:
                jobs[cid] = {
                    "id": cid,
                    "status": JobStatus.PENDING,
                    "attempts": 0,
                    "output_file": None,
                    "duration": c.get("duration"),
                    "score": c.get("score"),
                    "hook": c.get("hook"),
                    "reason": c.get("reason"),
                    "validation_result": None,
                    "error": None,
                    "last_attempt_at": None
                }
        state["jobs"] = jobs
        state["status"] = RunStatus.CANDIDATES_READY
        self.save_run(state)

    def update_job(
        self,
        run_id: str,
        clip_id: str,
        status: str,
        output_file: Optional[str] = None,
        duration: Optional[float] = None,
        validation: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        increment_attempt: bool = False
    ):
        state = self.load_run(run_id)
        if not state:
            return

        if clip_id not in state["jobs"]:
            state["jobs"][clip_id] = {"id": clip_id, "attempts": 0}

        job = state["jobs"][clip_id]
        job["status"] = status
        now = time.time()
        job["last_attempt_at"] = now

        if increment_attempt:
            job["attempts"] = job.get("attempts", 0) + 1
        if output_file:
            job["output_file"] = str(output_file)
        if duration is not None:
            job["duration"] = duration
        if validation:
            job["validation_result"] = validation
        if error:
            job["error"] = str(error)
        elif status == JobStatus.COMPLETED:
            job["error"] = None

        # Recompute batch metrics
        completed = sum(1 for j in state["jobs"].values() if j.get("status") == JobStatus.COMPLETED)
        failed = sum(1 for j in state["jobs"].values() if j.get("status") == JobStatus.FAILED)
        total = len(state["jobs"])

        state["metrics"]["completed_jobs"] = completed
        state["metrics"]["failed_jobs"] = failed

        if completed == total and total > 0:
            state["status"] = RunStatus.COMPLETED
        elif completed > 0 and (completed + failed == total):
            state["status"] = RunStatus.PARTIALLY_COMPLETED
        elif failed == total and total > 0:
            state["status"] = RunStatus.FAILED

        self.save_run(state)

    def list_runs(self) -> List[Dict[str, Any]]:
        runs = []
        if not self.runs_dir.exists():
            return []

        for p in self.runs_dir.iterdir():
            if p.is_dir() and (p / "state.json").exists():
                st = self.load_run(p.name)
                if st:
                    runs.append({
                        "run_id": st["run_id"],
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.get("created_at", 0))),
                        "status": st.get("status"),
                        "video_id": st.get("source", {}).get("video_id"),
                        "title": st.get("source", {}).get("title"),
                        "completed_jobs": st.get("metrics", {}).get("completed_jobs", 0),
                        "total_jobs": st.get("metrics", {}).get("total_jobs", 0)
                    })

        # Sort newest first
        runs.sort(key=lambda r: r["created_at"], reverse=True)
        return runs


# Global run manager instance
run_manager = RunManager()
