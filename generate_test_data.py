"""
Test Data Generator
====================
Generates fake .npy keypoint files to test train.py
before real video clips are available.

Creates:
    data/keypoints/
    └── ALGORITHM/
        ├── signer1_rep01.npy   # shape: (45, 75, 3)
        ├── signer1_rep02.npy
        └── ... (20 reps x 5 signers = 100 clips)
    └── VARIABLE/
        └── ...
    └── ... (however many test glosses you specify)

USAGE:
    # Generate 5 test glosses, 5 signers, 20 reps each
    python generate_test_data.py

    # Generate with custom settings
    python generate_test_data.py --glosses 10 --signers 5 --reps 20

    # Clean up generated test data
    python generate_test_data.py --clean
"""

import argparse
import shutil
from pathlib import Path
import numpy as np

REPO_ROOT     = Path(__file__).resolve().parent
KEYPOINTS_DIR = REPO_ROOT / "data" / "keypoints"

# A small subset of your real glosses for testing
TEST_GLOSSES = [
    "ALGORITHM", "VARIABLE", "FUNCTION",
    "LOOP", "ARRAY", "SORT", "SEARCH",
    "CLASS", "ERROR", "RECURSION"
]

NUM_JOINTS   = 75
IN_CHANNELS  = 3
MIN_FRAMES   = 30
MAX_FRAMES   = 60


def generate_clip(num_frames: int) -> np.ndarray:
    """
    Generates a random keypoint array of shape (frames, 75, 3).
    Values are in [0, 1] range like real MediaPipe output.
    """
    return np.random.rand(num_frames, NUM_JOINTS, IN_CHANNELS).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glosses",  type=int, default=5,
                        help="Number of glosses to generate (max 10)")
    parser.add_argument("--signers",  type=int, default=5)
    parser.add_argument("--reps",     type=int, default=20)
    parser.add_argument("--clean",    action="store_true",
                        help="Delete all generated test data and exit")
    args = parser.parse_args()

    # Clean mode
    if args.clean:
        if KEYPOINTS_DIR.exists():
            shutil.rmtree(KEYPOINTS_DIR)
            print(f"[CLEAN] Deleted {KEYPOINTS_DIR}")
        else:
            print("[CLEAN] Nothing to delete.")
        return

    glosses = TEST_GLOSSES[:args.glosses]
    total   = len(glosses) * args.signers * args.reps

    print(f"\n  Generating test keypoint data")
    print(f"  Glosses  : {len(glosses)}")
    print(f"  Signers  : {args.signers}")
    print(f"  Reps     : {args.reps}")
    print(f"  Total    : {total} clips\n")

    for gloss in glosses:
        gloss_dir = KEYPOINTS_DIR / gloss
        gloss_dir.mkdir(parents=True, exist_ok=True)

        for signer in range(1, args.signers + 1):
            for rep in range(1, args.reps + 1):
                filename = f"signer{signer}_rep{rep:02d}.npy"
                out_path = gloss_dir / filename

                if out_path.exists():
                    continue

                # Random frame count between MIN and MAX frames
                num_frames = np.random.randint(MIN_FRAMES, MAX_FRAMES + 1)
                clip       = generate_clip(num_frames)
                np.save(str(out_path), clip)

        print(f"  [DONE] {gloss} — {args.signers * args.reps} clips saved")

    print(f"\n  Test data ready at: {KEYPOINTS_DIR}")
    print(f"  Run: python train.py\n")


if __name__ == "__main__":
    main()