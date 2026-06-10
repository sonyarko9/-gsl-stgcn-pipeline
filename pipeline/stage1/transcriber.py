"""
Stage 1 — Real-Time Whisper ASR Transcriber
============================================
Continuously records audio in 4-second windows, filters silence,
transcribes with Whisper, and writes new text to
pipeline/stage2/live_text_buffer.txt for Stage 2 to consume.
"""

import whisper
import sounddevice as sd
import numpy as np
import time
from pathlib import Path

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
SAMPLE_RATE    = 16000
CHUNK_DURATION = 4.0
SILENCE_RMS    = 0.01

# Keywords that identify a headset/earpiece mic
PREFERRED_KEYWORDS = ["headset", "earphone", "earpiece", "headphone", "usb", "external"]

# Set to a device index number to force a specific mic, or None for auto-detection
FORCE_DEVICE_INDEX = None

_DIR        = Path(__file__).parent
BUFFER_PATH = _DIR.parent / "stage2" / "live_text_buffer.txt"

BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
BUFFER_PATH.write_text("", encoding="utf-8")   # clear stale content on startup


# ─────────────────────────────
# AUDIO DEVICE SETUP
# ─────────────────────────────
def test_device(index: int) -> bool:
    """
    Returns True if the device can actually be opened for recording.
    Bluetooth headsets that are detected but not connected will fail this test.
    """
    try:
        test = sd.rec(
            int(0.1 * SAMPLE_RATE),   # 0.1 second test recording
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=index
        )
        sd.wait()
        return True
    except Exception:
        return False


def get_input_device():
    """
    Device selection priority:
      1. FORCE_DEVICE_INDEX if set manually at the top of this file
      2. First preferred keyword match that passes the live device test
      3. Windows default input device as fallback
    Prints all available input devices on every startup for visibility.
    """
    devices = sd.query_devices()

    print("[INFO] Available input devices:", flush=True)
    input_devices = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"  [{i}] {dev['name']}", flush=True)
            input_devices.append((i, dev["name"].lower()))

    # ── Priority 1: manual override ──────────────────────────────────────────
    if FORCE_DEVICE_INDEX is not None:
        name = devices[FORCE_DEVICE_INDEX]["name"]
        print(f"[INFO] Using forced device [{FORCE_DEVICE_INDEX}]: {name}", flush=True)
        return FORCE_DEVICE_INDEX

    # ── Priority 2: auto-detect preferred device that is actually connected ──
    print("[INFO] Testing preferred devices...", flush=True)
    for idx, name in input_devices:
        for keyword in PREFERRED_KEYWORDS:
            if keyword in name:
                print(f"[INFO] Testing [{idx}]: {devices[idx]['name']} ...", flush=True)
                if test_device(idx):
                    print(f"[INFO] Selected working device [{idx}]: {devices[idx]['name']}", flush=True)
                    return idx
                else:
                    print(f"[SKIP] Device [{idx}] not available (not connected): {devices[idx]['name']}", flush=True)

    # ── Priority 3: fall back to Windows default ─────────────────────────────
    try:
        default = sd.query_devices(kind="input")
        print(f"[INFO] No preferred device available. Falling back to default: {default['name']}", flush=True)
    except Exception as e:
        print(f"[WARN] Could not query default device: {e}", flush=True)

    return None   # None = sounddevice uses whatever Windows has set as default


INPUT_DEVICE = get_input_device()


# ─────────────────────────────
# LOAD MODEL
# ─────────────────────────────
print("[INFO] Loading Whisper (tiny)...", flush=True)
model = whisper.load_model("tiny")
print("[INFO] Model ready.\n", flush=True)


# ─────────────────────────────
# AUDIO RECORDING
# ─────────────────────────────
def record_chunk() -> np.ndarray:
    audio = sd.rec(
        int(CHUNK_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=INPUT_DEVICE
    )
    sd.wait()
    return audio.flatten()


def is_silent(audio: np.ndarray) -> bool:
    return float(np.sqrt(np.mean(audio ** 2))) < SILENCE_RMS


# ─────────────────────────────
# TRANSCRIBE
# ─────────────────────────────
def transcribe(audio: np.ndarray) -> str:
    result = model.transcribe(
        audio,
        language="en",
        fp16=False,
        temperature=0.0,
        condition_on_previous_text=False
    )
    return result["text"].strip()


# ─────────────────────────────
# REAL-TIME LOOP
# ─────────────────────────────
print("[TRANSCRIBER] Running - speak into the microphone.\n", flush=True)

last_text = ""

while True:
    try:
        audio = record_chunk()

        if is_silent(audio):
            continue

        text = transcribe(audio)

        if not text or text == last_text:
            continue

        last_text = text
        print(f"[HEARD] {text}", flush=True)
        BUFFER_PATH.write_text(text, encoding="utf-8")

    except KeyboardInterrupt:
        print("\n[TRANSCRIBER] Stopped.")
        break
    except Exception as e:
        print(f"[TRANSCRIBER ERROR] {e}", flush=True)
        time.sleep(0.5)