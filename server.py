"""
NeuroChair backend: serves the trained SNN keyword spotter to the 3D web UI.

The browser captures microphone audio, detects when a word was spoken (simple
energy gate), and POSTs ~1.5 s of 16 kHz float32 PCM to /predict. This server:
  1. picks the loudest 1-second window (so the word is roughly centred, like
     the Speech Commands training clips),
  2. runs the exact same MFCC -> spike-train encoder used in training,
  3. runs the SNN on several independent Bernoulli spike samples, for the window
     and two copies shifted by +-80 ms, and averages the logits (rate coding is
     stochastic and word position varies -- averaging makes predictions stable),
  4. masks out keywords the wheelchair UI isn't listening for, and
  5. returns class probabilities + the hidden-layer spike rasters for display.

Usage:
    pip install -r requirements.txt
    python server.py                      # uses checkpoints/snn_checkpoint.pt
    python server.py --checkpoint path/to/snn_checkpoint.pt
Then open http://localhost:8000

Voice calibration (see personalize.py): the UI records a few clips per command into
recordings/<session>/, and /api/calibrate/train fine-tunes the SNN on them. The result is
saved as checkpoints/snn_personal.pt and loaded automatically on the next start.
"""

import argparse
import copy
import sys
import threading
import time
from pathlib import Path

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.speech_commands import MFCCSpikeEncoder, KEYWORDS, N_MFCC, SAMPLE_RATE, CLIP_SAMPLES  # noqa: E402
from snn.model import SNNClassifier  # noqa: E402
from personalize import calibrate  # noqa: E402

STATIC_DIR = ROOT / "static"
DEFAULT_CHECKPOINTS = [ROOT / "checkpoints" / "snn_checkpoint.pt"]
PERSONAL_CHECKPOINT = ROOT / "checkpoints" / "snn_personal.pt"
RECORDINGS_DIR = ROOT / "recordings"
COMMAND_WORDS = ["go", "stop", "left", "right", "up", "down", "on", "off"]
SHIFT_SAMPLES = int(0.08 * SAMPLE_RATE)

app = FastAPI(title="NeuroChair")
state = {"model": None, "base_model": None, "encoder": None, "loaded": False, "checkpoint": None,
         "personalized": False, "n_samples": 8, "session": None}
train_lock = threading.Lock()


def load_model(checkpoint: str | None, hidden_sizes: list[int]):
    model = SNNClassifier(in_features=N_MFCC, hidden_sizes=hidden_sizes, n_classes=len(KEYWORDS))
    candidates = [Path(checkpoint)] if checkpoint else DEFAULT_CHECKPOINTS
    for path in candidates:
        if path.is_file():
            model.load_state_dict(torch.load(path, map_location="cpu"))
            state.update(loaded=True, checkpoint=str(path))
            print(f"[neurochair] loaded checkpoint: {path}")
            break
    else:
        print("[neurochair] WARNING: no checkpoint found -- running an UNTRAINED model (random predictions).")
        print("             Put snn_checkpoint.pt in checkpoints/ (or pass --checkpoint <path>).")
    model.eval()
    state["base_model"] = model
    state["model"] = model
    state["encoder"] = MFCCSpikeEncoder(coding="rate")


def load_personal():
    if PERSONAL_CHECKPOINT.is_file() and state["loaded"]:
        personal = copy.deepcopy(state["base_model"])
        personal.load_state_dict(torch.load(PERSONAL_CHECKPOINT, map_location="cpu"))
        personal.eval()
        state.update(model=personal, personalized=True)
        print(f"[neurochair] loaded voice calibration: {PERSONAL_CHECKPOINT}")


def mask_for(words) -> torch.Tensor:
    return torch.tensor([w in set(words) for w in KEYWORDS])


def loudest_start(waveform: torch.Tensor) -> int:
    """Start index of the 1-second slice with the most energy."""
    if waveform.shape[0] <= CLIP_SAMPLES:
        return 0
    energy = waveform.pow(2)
    cumsum = torch.cat([torch.zeros(1), energy.cumsum(0)])
    hop = SAMPLE_RATE // 50  # 20 ms search step
    starts = torch.arange(0, waveform.shape[0] - CLIP_SAMPLES + 1, hop)
    window_energy = cumsum[starts + CLIP_SAMPLES] - cumsum[starts]
    return int(starts[window_energy.argmax()])


def window_at(waveform: torch.Tensor, start: int) -> torch.Tensor:
    """1-second slice starting at `start`, zero-padded wherever it runs off either end."""
    padded = torch.nn.functional.pad(waveform, (CLIP_SAMPLES, CLIP_SAMPLES))
    return padded[start + CLIP_SAMPLES:start + 2 * CLIP_SAMPLES]


def read_pcm(raw: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(raw), dtype=torch.float32)


@app.get("/api/status")
def status():
    return {
        "model_loaded": state["loaded"],
        "checkpoint": state["checkpoint"],
        "keywords": KEYWORDS,
        "hidden_sizes": [layer.synapse.out_features for layer in state["model"].hidden_layers],
        "personalized": state["personalized"],
    }


@app.post("/api/predict")
async def predict(request: Request):
    """
    Body: raw little-endian float32 PCM, mono, 16 kHz.
    Query: active=go,stop,left,...  (keywords the UI is listening for; others are masked out)
    """
    t0 = time.perf_counter()
    raw = await request.body()
    if len(raw) < 4 * SAMPLE_RATE // 4:
        return JSONResponse({"error": "need at least 0.25 s of audio"}, status_code=400)
    waveform = read_pcm(raw)
    best = loudest_start(waveform)

    active = request.query_params.get("active")
    mask = mask_for(active.split(",") if active else KEYWORDS)

    model, encoder = state["model"], state["encoder"]
    # centre window first, then +-80 ms shifted copies (test-time augmentation)
    windows = [window_at(waveform, best + d) for d in (0, -SHIFT_SAMPLES, SHIFT_SAMPLES)]
    mfccs = torch.stack([encoder.waveform_to_mfcc(w) for w in windows])  # (3, T, n_mfcc) in [0, 1]
    spikes_in = torch.bernoulli(mfccs.repeat_interleave(state["n_samples"], dim=0))

    with torch.no_grad():
        logits, hidden_spikes, readout_trace = model(spikes_in, return_spikes=True)
    logits = logits.mean(dim=0)
    probs_all = torch.softmax(logits, dim=0)
    masked = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked, dim=0)
    pred = int(probs.argmax())

    # --- activity / efficiency stats, for one spike sample (what a chip would actually run) ---
    in_spk = spikes_in[0]                    # (T, 20)
    h_spk = [h[0] for h in hidden_spikes]    # [(T, 128), (T, 128)]
    T = in_spk.shape[0]
    fan_outs = [layer.synapse.out_features for layer in model.hidden_layers] + [model.readout.synapse.out_features]
    layer_inputs = [in_spk] + h_spk
    # event-driven cost: each spike triggers one synaptic accumulate per outgoing connection
    sops = int(sum(s.sum().item() * fo for s, fo in zip(layer_inputs, fan_outs)))
    # dense (CNN/LSTM-style) cost: every input multiplies every weight at every step
    dense_macs = int(sum(s.shape[1] * fo * T for s, fo in zip(layer_inputs, fan_outs)))
    hidden_total = sum(int(h.sum()) for h in h_spk)
    hidden_capacity = sum(h.numel() for h in h_spk)

    def raster(s):
        t, n = torch.nonzero(s, as_tuple=True)
        return [t.tolist(), n.tolist()]

    return {
        "word": KEYWORDS[pred],
        "confidence": float(probs[pred]),
        "probs": {w: float(p) for w, p in zip(KEYWORDS, probs)},
        "probs_unmasked": {w: float(p) for w, p in zip(KEYWORDS, probs_all)},
        "raster": {
            "input": raster(in_spk),
            "hidden": [raster(h) for h in h_spk],
            "shape": {"T": T, "input": in_spk.shape[1], "hidden": [h.shape[1] for h in h_spk]},
        },
        "readout_trace": readout_trace.mean(dim=0).tolist(),  # (T, 10)
        "stats": {
            "hidden_spikes": hidden_total,
            "sparsity": 1 - hidden_total / hidden_capacity,
            "sops": sops,
            "dense_macs": dense_macs,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        },
        "model_loaded": state["loaded"],
        "personalized": state["personalized"],
    }


# ---------------- voice calibration ----------------

@app.post("/api/calibrate/start")
def calibrate_start():
    session = RECORDINGS_DIR / time.strftime("session_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    state["session"] = session
    return {"session": session.name, "words": COMMAND_WORDS}


@app.post("/api/calibrate/clip")
async def calibrate_clip(request: Request):
    """Save one recorded clip for `word`; report what the BASE model thinks it is."""
    word = request.query_params.get("word")
    if word not in COMMAND_WORDS or state["session"] is None:
        return JSONResponse({"error": "call /api/calibrate/start first, with a valid ?word="}, status_code=400)
    waveform = read_pcm(await request.body())
    clip = window_at(waveform, loudest_start(waveform))
    n = len(list(state["session"].glob(f"{word}_*.wav")))
    sf.write(state["session"] / f"{word}_{n:02d}.wav", clip.numpy(), SAMPLE_RATE)

    encoder = state["encoder"]
    x = torch.bernoulli(encoder.waveform_to_mfcc(clip).unsqueeze(0).expand(state["n_samples"], -1, -1).contiguous())
    with torch.no_grad():
        logits = state["base_model"](x).mean(0).masked_fill(~mask_for(COMMAND_WORDS), float("-inf"))
    heard = KEYWORDS[int(logits.argmax())]
    return {"saved": n + 1, "base_heard": heard, "base_correct": heard == word}


@app.post("/api/calibrate/train")
def calibrate_train():
    if state["session"] is None:
        return JSONResponse({"error": "no calibration session"}, status_code=400)
    if not train_lock.acquire(blocking=False):
        return JSONResponse({"error": "already training"}, status_code=409)
    try:
        clips = []
        for f in sorted(state["session"].glob("*.wav")):
            audio, _ = sf.read(f, dtype="float32")
            clips.append((torch.from_numpy(audio), f.stem.rsplit("_", 1)[0]))
        if len({w for _, w in clips}) < len(COMMAND_WORDS):
            return JSONResponse({"error": "record at least one clip per command"}, status_code=400)
        model, report = calibrate(state["base_model"], clips, state["encoder"], mask_for(COMMAND_WORDS))
        torch.save(model.state_dict(), PERSONAL_CHECKPOINT)
        state.update(model=model, personalized=True)
        return report
    finally:
        train_lock.release()


@app.post("/api/calibrate/reset")
def calibrate_reset():
    """Go back to the original trained model (recordings are kept on disk)."""
    PERSONAL_CHECKPOINT.unlink(missing_ok=True)
    state.update(model=state["base_model"], personalized=False)
    return {"personalized": False}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--samples", type=int, default=8, help="Bernoulli spike samples averaged per window")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base_only", action="store_true", help="ignore checkpoints/snn_personal.pt")
    args = parser.parse_args()
    state["n_samples"] = args.samples
    load_model(args.checkpoint, args.hidden_sizes)
    if not args.base_only:
        load_personal()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
