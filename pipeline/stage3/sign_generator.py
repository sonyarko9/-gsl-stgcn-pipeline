# pipeline/stage3/sign_generator.py
import torch
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from models.stgcn_attention import STGCNAttention
from models.graph import build_mediapipe_adjacency
from pipeline.stage3.gloss_to_pose import GlossToPoseMapper
from pipeline.stage3 import config


class SignGenerator:
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)

        A = build_mediapipe_adjacency()
        self.model = STGCNAttention(
            in_channels=config.IN_CHANNELS,
            num_classes=config.NUM_CLASSES,
            A=A,
            num_heads=config.NUM_HEADS,
            d_model=config.D_MODEL
        ).to(self.device)

        if model_path and Path(model_path).exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            print(f"[SignGenerator] Weights loaded from {model_path}")
        else:
            print("[SignGenerator] No weights file — model in untrained state (expected at this stage)")

        self.model.eval()
        self.mapper = GlossToPoseMapper(
            keypoints_dir=str(config.KEYPOINTS_DIR)
        )

    def generate(self, gloss_sequence: list) -> dict:
        results = {}
        for gloss in gloss_sequence:
            tensor = self.mapper.to_tensor(gloss)
            if tensor is None:
                results[gloss] = "NO_KEYPOINTS"
                continue
            tensor = tensor.to(self.device)
            with torch.no_grad():
                output = self.model(tensor)
            results[gloss] = output.cpu().numpy().tolist()
        return results

    def run_from_file(self, gloss_json_path: str) -> dict:
        with open(gloss_json_path) as f:
            data = json.load(f)
        glosses = data.get("glosses", [])
        print(f"[SignGenerator] Loaded glosses: {glosses}")
        return self.generate(glosses)