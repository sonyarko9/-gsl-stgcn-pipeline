"""
MediaPipe Holistic Keypoint Extractor
======================================
Converts recorded .mp4 clips → .npy keypoint files in data/keypoints/

FOLDER STRUCTURE EXPECTED:
    data/raw/
    └── COMPUTER/
        ├── signer1_rep01.mp4
        ├── signer1_rep02.mp4
        ├── ...
        └── signer5_rep20.mp4
    └── PROGRAM/
        └── ...

OUTPUT:
    data/keypoints/
    └── COMPUTER/
        ├── signer1_rep01.npy   # shape: (frames, 75, 3)
        ├── signer1_rep02.npy
        └── ...

Each .npy file = one clip = (frames, 75, 3)
  - 75 joints: 33 pose + 21 left hand + 21 right hand
  - 3 values per joint: x, y, visibility/confidence

USAGE:
    # Extract all signs
    python extract_keypoints.py

    # Extract a single sign (for testing)
    python extract_keypoints.py --gloss COMPUTER

    # Dry run (shows what would be processed, no files written)
    python extract_keypoints.py --dry-run

    # Show extraction report after run
    python extract_keypoints.py --report
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ── Repo root (script lives at project root) ──────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent
RAW_DIR      = REPO_ROOT / "data" / "raw"
KEYPOINTS_DIR = REPO_ROOT / "data" / "keypoints"
MIN_FRAMES   = 9       # must match config.MIN_FRAMES (STGCNLayer temporal kernel)
NUM_JOINTS   = 75      # 33 pose + 21 left hand + 21 right hand

# ── MediaPipe joint counts ────────────────────────────────────────────────────
N_POSE       = 33
N_HAND       = 21      # each hand


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_landmarks_from_frame(results) -> np.ndarray:
    """
    Flattens MediaPipe Holistic results for one frame into a (75, 3) array.
    Order: [pose(33), left_hand(21), right_hand(21)]
    Each row: [x, y, visibility_or_confidence]
    Missing landmarks (hand not detected) → zeros.
    """
    frame_kp = np.zeros((NUM_JOINTS, 3), dtype=np.float32)

    # --- Pose (landmarks 0–32) ---
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            frame_kp[i] = [lm.x, lm.y, lm.visibility]

    # --- Left hand (landmarks 33–53) ---
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            frame_kp[N_POSE + i] = [lm.x, lm.y, lm.presence if hasattr(lm, 'presence') else 1.0]

    # --- Right hand (landmarks 54–74) ---
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            frame_kp[N_POSE + N_HAND + i] = [lm.x, lm.y, lm.presence if hasattr(lm, 'presence') else 1.0]

    return frame_kp


def extract_clip(video_path: Path, holistic) -> np.ndarray | None:
    """
    Runs MediaPipe Holistic on every frame of a video clip.
    Returns array of shape (frames, 75, 3) or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    [ERROR] Cannot open video: {video_path.name}")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe requires RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        frame_kp = extract_landmarks_from_frame(results)
        frames.append(frame_kp)

    cap.release()

    if len(frames) == 0:
        print(f"    [ERROR] No frames extracted from: {video_path.name}")
        return None

    if len(frames) < MIN_FRAMES:
        print(f"    [WARN]  {video_path.name} has only {len(frames)} frames "
              f"(minimum {MIN_FRAMES}) — skipping")
        return None

    return np.array(frames, dtype=np.float32)   # (frames, 75, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────────────────────────

def process_gloss(gloss_dir: Path, dry_run: bool = False) -> dict:
    """
    Process all .mp4 clips for one gloss.
    Returns a stats dict for the report.
    """
    gloss = gloss_dir.name
    clips = sorted(gloss_dir.glob("*.mp4"))

    if not clips:
        print(f"  [{gloss}] No .mp4 files found — skipping")
        return {"gloss": gloss, "total": 0, "saved": 0, "skipped": 0, "errors": 0}

    out_dir = KEYPOINTS_DIR / gloss
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"gloss": gloss, "total": len(clips), "saved": 0, "skipped": 0, "errors": 0}

    print(f"\n  [{gloss}] {len(clips)} clip(s) found")

    # Initialise MediaPipe Holistic once per gloss (reused across clips)
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,           # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for clip_path in clips:
            out_path = out_dir / f"{clip_path.stem}.npy"

            # Skip already-extracted clips
            if out_path.exists():
                print(f"    [SKIP] {clip_path.name} → already extracted")
                stats["skipped"] += 1
                continue

            if dry_run:
                print(f"    [DRY]  Would extract: {clip_path.name} → {out_path.name}")
                stats["saved"] += 1
                continue

            keypoints = extract_clip(clip_path, holistic)

            if keypoints is None:
                stats["errors"] += 1
                continue

            np.save(str(out_path), keypoints)
            print(f"    [SAVE] {clip_path.name} → {out_path.name}  "
                  f"shape={keypoints.shape}  frames={keypoints.shape[0]}")
            stats["saved"] += 1

    return stats


def print_report(all_stats: list[dict], elapsed: float):
    """Print a summary table after extraction."""
    print("\n" + "="*55)
    print("  EXTRACTION REPORT")
    print("="*55)
    print(f"  {'Gloss':<20} {'Total':>6} {'Saved':>6} {'Skip':>6} {'Err':>6}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    total_clips = saved = skipped = errors = 0
    for s in all_stats:
        print(f"  {s['gloss']:<20} {s['total']:>6} {s['saved']:>6} "
              f"{s['skipped']:>6} {s['errors']:>6}")
        total_clips += s["total"]
        saved       += s["saved"]
        skipped     += s["skipped"]
        errors      += s["errors"]

    print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    print(f"  {'TOTAL':<20} {total_clips:>6} {saved:>6} {skipped:>6} {errors:>6}")
    print(f"\n  Time elapsed : {elapsed:.1f}s")
    print(f"  Output dir   : {KEYPOINTS_DIR}")

    if errors > 0:
        print(f"\n  ⚠️  {errors} clip(s) failed — check logs above")
    else:
        print(f"\n  ✅ Extraction complete — {saved} clip(s) saved")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic keypoints from GSL video clips"
    )
    parser.add_argument(
        "--gloss", type=str, default=None,
        help="Process a single gloss only (e.g. --gloss COMPUTER)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without writing any files"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print extraction report after run (always shown if errors occur)"
    )
    args = parser.parse_args()

    # ── Validate raw directory ────────────────────────────────────────────────
    if not RAW_DIR.exists():
        print(f"[ERROR] Raw video directory not found: {RAW_DIR}")
        print("        Create data/raw/<GLOSS>/<clips>.mp4 first")
        sys.exit(1)

    # ── Determine which glosses to process ───────────────────────────────────
    if args.gloss:
        gloss_dirs = [RAW_DIR / args.gloss]
        if not gloss_dirs[0].exists():
            print(f"[ERROR] Gloss folder not found: {gloss_dirs[0]}")
            sys.exit(1)
    else:
        gloss_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
        if not gloss_dirs:
            print(f"[ERROR] No gloss subdirectories found in {RAW_DIR}")
            sys.exit(1)

    mode = "DRY RUN" if args.dry_run else "EXTRACTION"
    print(f"\n{'='*55}")
    print(f"  GSL Keypoint Extractor — {mode}")
    print(f"{'='*55}")
    print(f"  Raw dir      : {RAW_DIR}")
    print(f"  Output dir   : {KEYPOINTS_DIR}")
    print(f"  Glosses      : {len(gloss_dirs)}")
    print(f"  Min frames   : {MIN_FRAMES}")
    print(f"  Joints       : {NUM_JOINTS} (33 pose + 21+21 hands)")

    # ── Run extraction ────────────────────────────────────────────────────────
    start = time.time()
    all_stats = []

    for gloss_dir in gloss_dirs:
        stats = process_gloss(gloss_dir, dry_run=args.dry_run)
        all_stats.append(stats)

    elapsed = time.time() - start

    # ── Report ────────────────────────────────────────────────────────────────
    has_errors = any(s["errors"] > 0 for s in all_stats)
    if args.report or has_errors:
        print_report(all_stats, elapsed)
    else:
        total_saved = sum(s["saved"] for s in all_stats)
        print(f"\n  ✅ Done — {total_saved} clip(s) processed in {elapsed:.1f}s")
        print(f"     Output: {KEYPOINTS_DIR}\n")


if __name__ == "__main__":
    main()