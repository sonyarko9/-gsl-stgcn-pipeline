"""
GSL-STGCN Pipeline — Stage 1 to 3 Test Suite
Run from project root:
    python test_pipeline.py
"""

import sys
import traceback

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ──────────────────────────────────────────────────────────
# TEST 1 — Adjacency Matrix
# ──────────────────────────────────────────────────────────
section("TEST 1 — Adjacency Matrix (models/graph.py)")
try:
    from models.graph import build_mediapipe_adjacency
    A = build_mediapipe_adjacency()

    assert A.shape == (75, 75), f"Wrong shape: {A.shape}"
    assert abs(A[0].sum() - 1.0) < 0.5, f"Row sum unexpected: {A[0].sum()}"

    print(f"  Shape         : {A.shape}")
    print(f"  Sample row sum: {A[0].sum():.4f}")
    print(f"  {PASS} — Adjacency matrix built correctly")
    results.append(("Test 1 — Adjacency Matrix", True))

except Exception as e:
    print(f"  {FAIL} — {e}")
    traceback.print_exc()
    results.append(("Test 1 — Adjacency Matrix", False))


# ──────────────────────────────────────────────────────────
# TEST 2 — Model Instantiation
# ──────────────────────────────────────────────────────────
section("TEST 2 — Model Instantiation (models/stgcn_attention.py)")
try:
    from models.stgcn_attention import STGCNAttention
    from models.graph import build_mediapipe_adjacency

    A = build_mediapipe_adjacency()
    model = STGCNAttention(
        in_channels=3,
        num_classes=50,
        A=A,
        num_heads=4,
        d_model=256
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params > 0, "Model has no parameters"

    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  {PASS} — Model instantiated correctly")
    results.append(("Test 2 — Model Instantiation", True))

except Exception as e:
    print(f"  {FAIL} — {e}")
    traceback.print_exc()
    results.append(("Test 2 — Model Instantiation", False))


# ──────────────────────────────────────────────────────────
# TEST 3 — Forward Pass with Dummy Data
# ──────────────────────────────────────────────────────────
section("TEST 3 — Forward Pass (batch=1, channels=3, frames=30, joints=75)")
try:
    import torch
    from models.stgcn_attention import STGCNAttention
    from models.graph import build_mediapipe_adjacency

    A = build_mediapipe_adjacency()
    model = STGCNAttention(
        in_channels=3,
        num_classes=50,
        A=A,
        num_heads=4,
        d_model=256
    )
    model.eval()

    dummy = torch.randn(1, 3, 30, 75)   # (batch, channels, frames, joints)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (1, 50), f"Wrong output shape: {out.shape}, expected (1, 50)"

    print(f"  Input shape : {list(dummy.shape)}")
    print(f"  Output shape: {list(out.shape)}  ← (batch=1, num_classes=50)")
    print(f"  {PASS} — Forward pass successful")
    results.append(("Test 3 — Forward Pass", True))

except Exception as e:
    print(f"  {FAIL} — {e}")
    traceback.print_exc()
    results.append(("Test 3 — Forward Pass", False))


# ──────────────────────────────────────────────────────────
# TEST 4 — SignGenerator End-to-End
# ──────────────────────────────────────────────────────────
section("TEST 4 — SignGenerator End-to-End (pipeline/stage3)")
try:
    from pipeline.stage3 import SignGenerator
    from pipeline.stage3 import config

    sg = SignGenerator()

    # Verify config values loaded correctly
    assert config.IN_CHANNELS  == 3,   f"IN_CHANNELS wrong: {config.IN_CHANNELS}"
    assert config.NUM_CLASSES  == 50,  f"NUM_CLASSES wrong: {config.NUM_CLASSES}"
    assert config.NUM_JOINTS   == 75,  f"NUM_JOINTS wrong: {config.NUM_JOINTS}"
    assert config.NUM_HEADS    == 4,   f"NUM_HEADS wrong: {config.NUM_HEADS}"
    assert config.D_MODEL      == 256, f"D_MODEL wrong: {config.D_MODEL}"

    print(f"  Model device  : {sg.device}")
    print(f"  Keypoints dir : {sg.mapper.keypoints_dir}")
    print(f"  IN_CHANNELS   : {config.IN_CHANNELS}")
    print(f"  NUM_CLASSES   : {config.NUM_CLASSES}")
    print(f"  NUM_JOINTS    : {config.NUM_JOINTS}")
    print(f"  NUM_HEADS     : {config.NUM_HEADS}")
    print(f"  D_MODEL       : {config.D_MODEL}")
    print(f"  {PASS} — SignGenerator ready")
    results.append(("Test 4 — SignGenerator", True))

except Exception as e:
    print(f"  {FAIL} — {e}")
    traceback.print_exc()
    results.append(("Test 4 — SignGenerator", False))


# ──────────────────────────────────────────────────────────
# TEST 5 — Dummy Gloss Inference (no real keypoints needed)
# ──────────────────────────────────────────────────────────
section("TEST 5 — Dummy Gloss Inference via SignGenerator")
try:
    import torch
    import numpy as np
    from pipeline.stage3 import SignGenerator
    from pipeline.stage3 import config

    sg = SignGenerator()

    # Manually inject a fake .npy file into the mapper cache
    # so we can test inference without real recorded data
    fake_gloss    = "TEST_SIGN"
    fake_keypoints = np.random.rand(30, 75, 3).astype(np.float32)  # (frames, joints, channels)
    sg.mapper._cache[fake_gloss] = fake_keypoints

    result = sg.generate([fake_gloss, "MISSING_SIGN"])

    assert fake_gloss in result,          "TEST_SIGN not in results"
    assert result[fake_gloss] is not None, "TEST_SIGN result is None"
    assert result["MISSING_SIGN"] == "NO_KEYPOINTS", "Missing gloss not handled"
    assert len(result[fake_gloss][0]) == config.NUM_CLASSES, \
        f"Output length {len(result[fake_gloss][0])} != NUM_CLASSES {config.NUM_CLASSES}"

    print(f"  Glosses tested    : {list(result.keys())}")
    print(f"  TEST_SIGN output  : list of {len(result[fake_gloss][0])} class scores ✓")
    print(f"  MISSING_SIGN      : '{result['MISSING_SIGN']}' ✓")
    print(f"  {PASS} — Inference pipeline working end-to-end")
    results.append(("Test 5 — Dummy Inference", True))

except Exception as e:
    print(f"  {FAIL} — {e}")
    traceback.print_exc()
    results.append(("Test 5 — Dummy Inference", False))


# ──────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────
section("SUMMARY")
passed = sum(1 for _, ok in results if ok)
total  = len(results)

for name, ok in results:
    status = PASS if ok else FAIL
    print(f"  {status}  {name}")

print(f"\n  Result: {passed}/{total} tests passed")

if passed == total:
    print("\n  🎉 All systems go — Stage 3 architecture is wired correctly.")
    print("     Next step: MediaPipe extraction script → populate data/keypoints/")
else:
    print("\n  ⚠️  Fix failing tests before proceeding.")

print()