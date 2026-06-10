"""
ST-GCN Attention Training Script
==================================
Trains the STGCNAttention model on extracted keypoint data.

EXPECTED DATA STRUCTURE:
    data/keypoints/
    └── ALGORITHM/
        ├── signer1_rep01.npy   # shape: (frames, 75, 3)
        ├── signer1_rep02.npy
        └── ...
    └── VARIABLE/
        └── ...

OUTPUT:
    models/stgcn_attention_trained.pth   <- trained model checkpoint
    models/label_map.json                <- gloss -> class index mapping

USAGE:
    python train.py
    python train.py --epochs 50 --batch-size 16
    python train.py --resume models/stgcn_attention_trained.pth
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parent
KEYPOINTS_DIR = REPO_ROOT / "data" / "keypoints"
MODELS_DIR    = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODELS_DIR / "stgcn_attention_trained.pth"
LABEL_MAP_PATH  = MODELS_DIR / "label_map.json"

# ── Config ────────────────────────────────────────────────────────────────────
NUM_JOINTS    = 75       # 33 pose + 21 left hand + 21 right hand
IN_CHANNELS   = 3        # x, y, visibility
MAX_FRAMES    = 60       # pad/truncate all clips to this length
NUM_HEADS     = 4
D_MODEL       = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
TRAIN_SPLIT   = 0.8      # 80% train, 20% validation


# ─────────────────────────────────────────────────────────────────────────────
# ADJACENCY MATRIX (75 joints: 33 pose + 21 left hand + 21 right hand)
# ─────────────────────────────────────────────────────────────────────────────

def build_adjacency_matrix(num_joints: int = 75) -> np.ndarray:
    """
    Builds a (75, 75) adjacency matrix for MediaPipe Holistic joints.
    Connections follow the actual human body skeleton structure.
    """
    A = np.zeros((num_joints, num_joints), dtype=np.float32)

    def connect(i, j):
        A[i][j] = 1.0
        A[j][i] = 1.0

    # ── Pose connections (joints 0-32, MediaPipe Pose) ────────────────────────
    pose_edges = [
        (0, 1), (1, 2), (2, 3), (3, 7),          # left eye
        (0, 4), (4, 5), (5, 6), (6, 8),           # right eye
        (9, 10),                                   # mouth
        (11, 12),                                  # shoulders
        (11, 13), (13, 15),                        # left arm
        (12, 14), (14, 16),                        # right arm
        (15, 17), (15, 19), (15, 21),              # left hand root
        (16, 18), (16, 20), (16, 22),              # right hand root
        (11, 23), (12, 24),                        # torso to hips
        (23, 24),                                  # hips
        (23, 25), (25, 27), (27, 29), (29, 31),   # left leg
        (24, 26), (26, 28), (28, 30), (30, 32),   # right leg
    ]
    for i, j in pose_edges:
        connect(i, j)

    # ── Left hand connections (joints 33-53, MediaPipe Hand) ─────────────────
    # Wrist = 33, then 4 fingers x 4 joints + thumb x 4
    lh = 33  # left hand offset
    hand_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # index
        (0, 9), (9, 10), (10, 11), (11, 12),       # middle
        (0, 13), (13, 14), (14, 15), (15, 16),     # ring
        (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
        (5, 9), (9, 13), (13, 17),                 # palm knuckles
    ]
    for i, j in hand_edges:
        connect(lh + i, lh + j)

    # ── Right hand connections (joints 54-74, MediaPipe Hand) ────────────────
    rh = 54  # right hand offset
    for i, j in hand_edges:
        connect(rh + i, rh + j)

    # ── Connect wrists to pose ────────────────────────────────────────────────
    connect(15, 33)   # pose left wrist  -> left hand wrist
    connect(16, 54)   # pose right wrist -> right hand wrist

    # ── Self-connections (each joint attends to itself) ───────────────────────
    for i in range(num_joints):
        A[i][i] = 1.0

    # ── Row-normalise ─────────────────────────────────────────────────────────
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0   # avoid division by zero
    A = A / row_sum

    return A


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class GlossDataset(Dataset):
    """
    Loads all .npy keypoint files from data/keypoints/<GLOSS>/*.npy
    Pads or truncates every clip to MAX_FRAMES.
    Returns tensors of shape (IN_CHANNELS, MAX_FRAMES, NUM_JOINTS).
    """

    def __init__(self, keypoints_dir: Path, label_map: dict, max_frames: int = MAX_FRAMES):
        self.samples   = []   # list of (npy_path, class_index)
        self.max_frames = max_frames

        for gloss, class_idx in label_map.items():
            gloss_dir = keypoints_dir / gloss
            if not gloss_dir.exists():
                print(f"[WARN] No keypoints folder for gloss: {gloss}")
                continue
            clips = sorted(gloss_dir.glob("*.npy"))
            if not clips:
                print(f"[WARN] No .npy files found for gloss: {gloss}")
                continue
            for clip_path in clips:
                self.samples.append((clip_path, class_idx))

        print(f"[DATASET] {len(self.samples)} clips loaded across {len(label_map)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_path, label = self.samples[idx]

        # Load keypoints: (frames, 75, 3)
        kp = np.load(str(clip_path)).astype(np.float32)

        # Pad or truncate to MAX_FRAMES
        T = kp.shape[0]
        if T < self.max_frames:
            pad = np.zeros((self.max_frames - T, NUM_JOINTS, IN_CHANNELS), dtype=np.float32)
            kp = np.concatenate([kp, pad], axis=0)
        else:
            kp = kp[:self.max_frames]

        # Reshape to (IN_CHANNELS, MAX_FRAMES, NUM_JOINTS) for ST-GCN
        # Original: (frames, joints, channels) -> (channels, frames, joints)
        kp = kp.transpose(2, 0, 1)

        return torch.tensor(kp), torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for keypoints, labels in loader:
        keypoints = keypoints.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        outputs = model(keypoints)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * keypoints.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += keypoints.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for keypoints, labels in loader:
            keypoints = keypoints.to(device)
            labels    = labels.to(device)

            outputs    = model(keypoints)
            loss       = criterion(outputs, labels)
            total_loss += loss.item() * keypoints.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += keypoints.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train STGCNAttention on GSL keypoints")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Device: {device}")
    if device.type == "cuda":
        print(f"[TRAIN] GPU: {torch.cuda.get_device_name(0)}")

    # ── Build label map from keypoints directory ──────────────────────────────
    gloss_dirs = sorted([d for d in KEYPOINTS_DIR.iterdir() if d.is_dir()])
    if not gloss_dirs:
        print(f"[ERROR] No keypoint folders found in {KEYPOINTS_DIR}")
        print("        Run extract_keypoints.py first.")
        return

    label_map = {d.name: idx for idx, d in enumerate(gloss_dirs)}
    num_classes = len(label_map)
    print(f"[TRAIN] Classes: {num_classes} — {list(label_map.keys())}")

    # Save label map so Stage 3 can use it for inference
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)
    print(f"[TRAIN] Label map saved to {LABEL_MAP_PATH}")

    # ── Build adjacency matrix ────────────────────────────────────────────────
    A = build_adjacency_matrix(NUM_JOINTS)
    print(f"[TRAIN] Adjacency matrix shape: {A.shape}")

    # ── Model ─────────────────────────────────────────────────────────────────
    # Import here so this script works standalone
    import sys
    sys.path.insert(0, str(MODELS_DIR))
    from stgcn_attention import STGCNAttention

    model = STGCNAttention(
        in_channels=IN_CHANNELS,
        num_classes=num_classes,
        A=A,
        num_heads=NUM_HEADS,
        d_model=D_MODEL
    ).to(device)

    print(f"[TRAIN] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch  = checkpoint.get("epoch", 0)
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"[TRAIN] Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.1%}")

    # ── Dataset and loaders ───────────────────────────────────────────────────
    dataset = GlossDataset(KEYPOINTS_DIR, label_map)

    if len(dataset) == 0:
        print("[ERROR] Dataset is empty. Check your keypoints directory.")
        return

    train_size = int(len(dataset) * TRAIN_SPLIT)
    val_size   = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"[TRAIN] Train: {len(train_set)} | Val: {len(val_set)}")

    # ── Optimiser and loss ────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[TRAIN] Starting training for {args.epochs} epochs\n")
    print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*8}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"  {epoch+1:>6}  {train_loss:>10.4f}  {train_acc:>8.1%}  "
              f"{val_loss:>8.4f}  {val_acc:>6.1%}  {lr:>8.6f}  ({elapsed:.1f}s)")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":           epoch + 1,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_acc":    best_val_acc,
                "label_map":       label_map,
                "num_classes":     num_classes,
                "in_channels":     IN_CHANNELS,
                "num_joints":      NUM_JOINTS,
                "max_frames":      MAX_FRAMES,
                "d_model":         D_MODEL,
                "num_heads":       NUM_HEADS,
            }, CHECKPOINT_PATH)
            print(f"  --> New best saved: {best_val_acc:.1%}")

    print(f"\n[TRAIN] Done. Best val accuracy: {best_val_acc:.1%}")
    print(f"[TRAIN] Checkpoint saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()