"""
GSL Real-Time Pipeline - Entry Point
=====================================
Launches Stage 1 (Transcriber) and Stage 2 (Gloss Mapper) as
separate subprocesses and monitors them. Output from both stages
is forwarded live to this terminal.

Run from project root:
    python pipeline/main.py
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path
import logging

# ─────────────────────────────
# PATH SETUP
# ─────────────────────────────
BASE      = Path(__file__).resolve().parent        # pipeline/
REPO_ROOT = BASE.parent                            # project root
STAGE1    = BASE / "stage1" / "transcriber.py"
STAGE2    = BASE / "stage2" / "gloss_mapper.py"

# Add pipeline/ to path so logging_setup can be found
sys.path.insert(0, str(BASE))

# ─────────────────────────────
# LOGGING SETUP
# ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────
# VALIDATE STAGE FILES EXIST
# ─────────────────────────────
for label, path in [("Stage 1 transcriber", STAGE1), ("Stage 2 gloss mapper", STAGE2)]:
    if not path.exists():
        logger.error(f"[Pipeline] {label} not found at: {path}")
        sys.exit(1)

# ─────────────────────────────
# SUBPROCESS ENVIRONMENT
# ─────────────────────────────
SUB_ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "PYTHONPATH": str(REPO_ROOT),   # ensures all pipeline imports resolve correctly
}

# ─────────────────────────────
# OUTPUT STREAMING
# ─────────────────────────────
def stream_output(process, label: str):
    """Forward subprocess stdout to the logger with a stage label."""
    for line in iter(process.stdout.readline, b""):
        try:
            text = line.decode("utf-8", errors="replace").rstrip()
        except Exception:
            text = "<un-decodable-bytes>"
        if text:
            logger.info("[%s] %s", label, text)

# ─────────────────────────────
# STARTUP BANNER
# ─────────────────────────────
logger.info("=" * 50)
logger.info("  GSL REAL-TIME PIPELINE STARTING")
logger.info("=" * 50)
logger.info("[Pipeline] Repo root : %s", REPO_ROOT)
logger.info("[Pipeline] Stage 1   : %s", STAGE1)
logger.info("[Pipeline] Stage 2   : %s", STAGE2)
logger.info("=" * 50)

# ─────────────────────────────
# LAUNCH SUBPROCESSES
# ─────────────────────────────
logger.info("[Pipeline] Starting Transcriber (Stage 1)...")
transcriber = subprocess.Popen(
    [sys.executable, "-u", str(STAGE1)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=SUB_ENV
)
threading.Thread(
    target=stream_output,
    args=(transcriber, "Stage1"),
    daemon=True
).start()

time.sleep(1)

logger.info("[Pipeline] Starting Gloss Mapper (Stage 2)...")
mapper = subprocess.Popen(
    [sys.executable, "-u", str(STAGE2)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=SUB_ENV
)
threading.Thread(
    target=stream_output,
    args=(mapper, "Stage2"),
    daemon=True
).start()

logger.info("[Pipeline] Both stages running.")
logger.info("[Pipeline] Press Ctrl+C to stop.")
logger.info("=" * 50)

# ─────────────────────────────
# WATCHDOG LOOP
# ─────────────────────────────
try:
    while True:
        if transcriber.poll() is not None:
            logger.error("[ERROR] Transcriber (Stage 1) stopped unexpectedly.")
            logger.error("[ERROR] Check Stage 1 output above for the cause.")
            break
        if mapper.poll() is not None:
            logger.error("[ERROR] Gloss Mapper (Stage 2) stopped unexpectedly.")
            logger.error("[ERROR] Check Stage 2 output above for the cause.")
            break
        time.sleep(1)

except KeyboardInterrupt:
    logger.info("[Pipeline] Ctrl+C received — stopping all stages...")

finally:
    for label, proc in [("Stage 1", transcriber), ("Stage 2", mapper)]:
        if proc.poll() is None:          # only terminate if still running
            proc.terminate()
            try:
                proc.wait(timeout=5)
                logger.info("[Pipeline] %s stopped cleanly.", label)
            except subprocess.TimeoutExpired:
                proc.kill()
                logger.warning("[Pipeline] %s force-killed after timeout.", label)

    logger.info("=" * 50)
    logger.info("  PIPELINE STOPPED")
    logger.info("=" * 50)