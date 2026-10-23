import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

from src.config import config

LOGS_DIR = config.STORAGE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MAIN_LOG_FILE = LOGS_DIR / "ai_shorts.log"


class PipelineLogger:
    """
    Structured logger for the AI Shorts Generator.
    Logs human-friendly messages to console and structured timestamped traces to local files.
    """

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or "global"
        self.logger = logging.getLogger(f"shorts.{self.run_id}")
        self.logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers if re-instantiated
        if not self.logger.handlers:
            # 1. Main rotating file handler
            file_formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] [run:%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            fh = logging.FileHandler(MAIN_LOG_FILE, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(file_formatter)
            self.logger.addHandler(fh)

            # 2. Console stream handler (INFO and above)
            ch = logging.StreamHandler(sys.stderr)
            ch.setLevel(logging.INFO)
            console_formatter = logging.Formatter(fmt="[%(levelname)s] %(message)s")
            ch.setFormatter(console_formatter)
            self.logger.addHandler(ch)

    def attach_run_file(self, run_dir: Path):
        """Attaches a run-specific log file handler."""
        run_log = run_dir / "run.log"
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        rfh = logging.FileHandler(run_log, encoding="utf-8")
        rfh.setLevel(logging.DEBUG)
        rfh.setFormatter(file_formatter)
        self.logger.addHandler(rfh)

    def info(self, msg: str):
        self.logger.info(self._sanitize(msg))

    def debug(self, msg: str):
        self.logger.debug(self._sanitize(msg))

    def warning(self, msg: str):
        self.logger.warning(self._sanitize(msg))

    def error(self, msg: str, exc_info: bool = False):
        self.logger.error(self._sanitize(msg), exc_info=exc_info)

    def stage(self, stage_name: str, status: str, message: str = ""):
        detail = f" - {message}" if message else ""
        self.logger.info(f"STAGE [{stage_name}] -> {status.upper()}{detail}")

    def cache_event(self, key: str, hit: bool, details: str = ""):
        state = "HIT" if hit else "MISS"
        detail = f" ({details})" if details else ""
        self.logger.debug(f"CACHE {state}: {key}{detail}")

    def retry_event(self, job_id: str, attempt: int, max_attempts: int, error: str):
        self.logger.warning(
            f"RETRY job '{job_id}' (attempt {attempt}/{max_attempts}): {self._sanitize(error)}"
        )

    def validation_event(self, clip_id: str, passed: bool, metrics: Optional[Dict[str, Any]] = None):
        status = "PASSED" if passed else "FAILED"
        metric_str = f" | {metrics}" if metrics else ""
        self.logger.info(f"VALIDATION {clip_id}: {status}{metric_str}")

    @staticmethod
    def _sanitize(text: str) -> str:
        """Sanitizes potential sensitive tokens or key patterns."""
        if not isinstance(text, str):
            text = str(text)
        # Redact generic authorization or token keys
        import re
        text = re.sub(r'(api[_-]?key|token|auth)=[\w\.-]+', r'\1=[REDACTED]', text, flags=re.IGNORECASE)
        return text


# Global shared logger instance
default_logger = PipelineLogger("system")
