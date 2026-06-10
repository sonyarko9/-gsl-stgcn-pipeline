# pipeline/stage3/gloss_to_pose.py
import numpy as np
from pathlib import Path


class GlossToPoseMapper:
    """
    Maps a gloss token (e.g. "COMPUTER") to a keypoint tensor
    loaded from data/keypoints/<GLOSS>.npy
    Shape returned: (1, 3, frames, 75) — ready for STGCNAttention
    """

    def __init__(self, keypoints_dir: str = "data/keypoints"):
        self.keypoints_dir = Path(keypoints_dir)
        self._cache = {}

    def load(self, gloss: str) -> np.ndarray:
        """Load keypoint array for a single gloss. Returns None if not found."""
        if gloss in self._cache:
            return self._cache[gloss]

        path = self.keypoints_dir / f"{gloss}.npy"
        if not path.exists():
            print(f"[GlossToPoseMapper] WARNING: No keypoints for '{gloss}' at {path}")
            return None

        data = np.load(str(path))   # expected shape: (frames, 75, 3)
        self._cache[gloss] = data
        return data

    def to_tensor(self, gloss: str):
        """Returns torch tensor (1, 3, frames, 75) or None if missing."""
        import torch
        data = self.load(gloss)
        if data is None:
            return None

        # (frames, 75, 3) → (1, 3, frames, 75)
        tensor = torch.tensor(data, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor

    def batch(self, gloss_sequence: list) -> dict:
        """Run to_tensor for each gloss. Returns dict of gloss → tensor."""
        return {g: self.to_tensor(g) for g in gloss_sequence}