"""Stage 3 configuration."""
from __future__ import annotations

import os
from pathlib import Path

# --- Repo root ---
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Paths (env-overridable) ---
STAGE2_OUTPUT   = Path(os.environ.get("STAGE2_OUTPUT",   str(REPO_ROOT / "pipeline" / "stage3" / "gloss_output.json")))
LIVE_BUFFER     = Path(os.environ.get("LIVE_BUFFER",     str(REPO_ROOT / "pipeline" / "stage3" / "live_gloss_buffer.json")))
KEYPOINTS_DIR   = Path(os.environ.get("KEYPOINTS_DIR",   str(REPO_ROOT / "data" / "keypoints")))
MODEL_WEIGHTS   = Path(os.environ.get("MODEL_WEIGHTS",   str(REPO_ROOT / "models" / "stgcn_weights.pth")))
OUTPUT_SKELETON = Path(os.environ.get("OUTPUT_SKELETON", str(REPO_ROOT / "result" / "skeleton_output.json")))

# --- Model hyperparameters ---
IN_CHANNELS   = 3      # x, y, visibility per joint
NUM_CLASSES   = 50     # 50 GSL signs in dataset
NUM_JOINTS    = 75     # 33 pose + 21 left hand + 21 right hand
NUM_HEADS     = 4      # attention heads (proposal spec)
D_MODEL       = 256    # must match ST-GCN final layer output
DROPOUT       = 0.5

# --- Data collection spec ---
NUM_SIGNERS   = 5
REPS_PER_SIGN = 20
MIN_FRAMES    = 9      # temporal kernel size minimum

# --- Backwards-compatible string forms ---
STAGE2_OUTPUT_STR   = str(STAGE2_OUTPUT)
LIVE_BUFFER_STR     = str(LIVE_BUFFER)
KEYPOINTS_DIR_STR   = str(KEYPOINTS_DIR)
MODEL_WEIGHTS_STR   = str(MODEL_WEIGHTS)
OUTPUT_SKELETON_STR = str(OUTPUT_SKELETON)