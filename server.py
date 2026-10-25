"""
NeuroChair backend: serves keyword-spotting models to the 3D web UI.

Every checkpoint in checkpoints/ can be selected from the UI: the spiking network (SNN)
and the non-spiking CNN and LSTM baselines from snn-keyword-spotter. The model type and
layer sizes are read from the weights, so a new .pt file only has to be dropped in.

The browser captures microphone audio, detects when a word was spoken (simple energy gate),
and POSTs ~1.5 s of 16 kHz float32 PCM to /api/predict. This server:
  1. picks the loudest 1-second window (so the word is roughly centred, like the
     Speech Commands training clips),
  2. computes the same scaled MFCCs used in training,
  3. classifies that window and two copies shifted by +-80 ms and averages the logits;
     the SNN sees several independent Bernoulli spike samples of each window (rate coding
     is stochastic), the CNN/LSTM read the MFCCs directly,
  4. masks out keywords the wheelchair UI isn't listening for, and
  5. returns class probabilities, plus the hidden-layer spike rasters for the SNN.

Usage:
    pip install -r requirements.txt
    python server.py                          # starts with checkpoints/snn_checkpoint.pt
    python server.py --checkpoint cnn_checkpoint.pt
Then open http://localhost:8737

Voice calibration (see personalize.py): the UI records a few clips per command into
recordings/<session>/, and /api/calibrate/train fine-tunes the active model on them. The
result is saved as checkpoints/<model>_personal.pt and shows up in the model list.
"""

import argparse
import json
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.speech_commands import MFCCSpikeEncoder, KEYWORDS, N_MFCC, N_TIME_STEPS, SAMPLE_RATE, CLIP_SAMPLES  # noqa: E402
from snn.model import SNNClassifier  # noqa: E402
from snn.baselines import CNNBaseline, LSTMBaseline  # noqa: E402
from personalize import calibrate  # noqa: E402

STATIC_DIR = ROOT / "static"
CKPT_DIR = ROOT / "checkpoints"
DEFAULT_CHECKPOINT = "snn_checkpoint.pt"
RECORDINGS_DIR = ROOT / "recordings"
CORRECTIONS_DIR = RECORDINGS_DIR / "corrections"
COMMAND_WORDS = ["go", "stop", "left", "right", "up", "down", "on", "off"]
SHIFT_SAMPLES = int(0.08 * SAMPLE_RATE)
KIND_NAMES = {"snn": "SNN", "cnn": "CNN", "lstm": "LSTM"}

@asynccontextmanager
async def lifespan(app):
    # Also initialize when launched with `python -m uvicorn server:app`.
    if state["model"] is None:
        initialize_model()
    yield


app = FastAPI(title="NeuroChair", lifespan=lifespan)
state = {"model": None, "kind": "snn", "name": None, "encoder": MFCCSpikeEncoder(coding="rate"),
         "n_samples": 8, "session": None}
train_lock = threading.Lock()


# ---------------- models ----------------

def build_model(sd: dict):
    """Rebuild a model from its state dict alone. Returns (model, kind)."""
    if "hidden_layers.0.synapse.weight" in sd:
        return SNNClassifier.from_state_dict(sd), "snn"
    if "lstm.weight_ih_l0" in sd:
        model, kind = LSTMBaseline(n_mfcc=sd["lstm.weight_ih_l0"].shape[1], n_classes=sd["classifier.weight"].shape[0],
                                   hidden_size=sd["lstm.weight_hh_l0"].shape[1]), "lstm"
    elif "net.0.weight" in sd:
        model, kind = CNNBaseline(n_mfcc=sd["net.0.weight"].shape[1], n_time_steps=N_TIME_STEPS,
                                  n_classes=sd["classifier.weight"].shape[0]), "cnn"
    else:
        raise ValueError("not an SNN, CNN or LSTM checkpoint")
    model.load_state_dict(sd)
    return model, kind


def meta(path: Path) -> dict:
    """Optional sidecar <name>.json with label / test_accuracy / note / base."""
    side = path.with_suffix(".json")
    return json.loads(side.read_text()) if side.is_file() else {}


_describe_cache = {}


def describe(path: Path) -> dict:
    key = (path.name, path.stat().st_mtime)
    if key not in _describe_cache:
        model, kind = build_model(torch.load(path, map_location="cpu", weights_only=True))
        m = meta(path)
        _describe_cache[key] = {
            "name": path.name, "kind": kind, "label": m.get("label", f"{KIND_NAMES[kind]} ({path.stem})"),
            "test_accuracy": m.get("test_accuracy"), "note": m.get("note", ""),
            "params": sum(p.numel() for p in model.parameters()), "base": m.get("base"),
        }
    return _describe_cache[key]


def checkpoint_path(name: str) -> Path:
    """Resolve a file name inside checkpoints/ (no directories, so no path traversal)."""
    path = CKPT_DIR / Path(name).name
    if path.suffix != ".pt" or not path.is_file():
        raise FileNotFoundError(name)
    return path


def load_model(path: Path):
    model, kind = build_model(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    state.update(model=model, kind=kind, name=path.name)
    print(f"[neurochair] active model: {path.name} ({KIND_NAMES[kind]})")


def initialize_model(checkpoint: str | None = None, base_only: bool = False):
    start = checkpoint or DEFAULT_CHECKPOINT
    personal = CKPT_DIR / f"{Path(start).stem}_personal.pt"
    if not checkpoint and not base_only and personal.is_file():
        start = personal.name
    try:
        load_model(checkpoint_path(start))
    except (FileNotFoundError, ValueError, RuntimeError):
        state["model"] = SNNClassifier(in_features=N_MFCC, hidden_sizes=[128, 128], n_classes=len(KEYWORDS)).eval()
        print(f"[neurochair] WARNING: could not load {start} -- running an UNTRAINED SNN (random predictions).")
        print("             Put a checkpoint in checkpoints/ or choose one in the UI.")


def mask_for(words) -> torch.Tensor:
    return torch.tensor([w in set(words) for w in KEYWORDS])


def dense_macs(model, kind: str, T: int = N_TIME_STEPS) -> int:
    """Multiply-accumulates of one forward pass of a non-spiking baseline."""
    if kind == "lstm":
        H, F = model.lstm.hidden_size, model.lstm.input_size
        return T * 4 * H * (F + H) + model.classifier.in_features * model.classifier.out_features
    L, macs = T, 0
    for m in model.net:
        if isinstance(m, torch.nn.Conv1d):
            macs += m.out_channels * m.in_channels * m.kernel_size[0] * L
        elif isinstance(m, torch.nn.MaxPool1d):
            L //= m.kernel_size
    return macs + model.classifier.in_features * model.classifier.out_features


@torch.no_grad()
def logits_for(model, kind: str, mfccs: torch.Tensor) -> torch.Tensor:
    """mfccs: (W, T, F) windows -> mean logits (n_classes,). The SNN averages K spike samples per window."""
    if kind == "snn":
        mfccs = torch.bernoulli(mfccs.repeat_interleave(state["n_samples"], dim=0))
    return model(mfccs).mean(0)


# ---------------- audio helpers ----------------

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


def save_clip(folder: Path, word: str, waveform: torch.Tensor) -> torch.Tensor:
    folder.mkdir(parents=True, exist_ok=True)
    clip = window_at(waveform, loudest_start(waveform))
    n = len(list(folder.glob(f"{word}_*.wav")))
    sf.write(folder / f"{word}_{n:02d}.wav", clip.numpy(), SAMPLE_RATE)
    return clip


def n_corrections() -> int:
    return len(list(CORRECTIONS_DIR.glob("*.wav")))


# ---------------- API ----------------

@app.get("/api/status")
def status():
    info = describe(CKPT_DIR / state["name"]) if state["name"] and (CKPT_DIR / state["name"]).is_file() else {}
    model = state["model"]
    return {
        "model_loaded": state["name"] is not None,
        "checkpoint": state["name"],
        "kind": state["kind"],
        "label": info.get("label", state["name"] or "untrained SNN"),
        "base": info.get("base"),
        "base_label": describe(CKPT_DIR / info["base"])["label"] if info.get("base") and (CKPT_DIR / info["base"]).is_file() else None,
        "keywords": KEYWORDS,
        "hidden_sizes": [layer.synapse.out_features for layer in model.hidden_layers] if state["kind"] == "snn" else None,
        "personalized": bool(info.get("base")),
        "corrections": n_corrections(),
    }


@app.get("/api/models")
def models():
    """Every loadable checkpoint in checkpoints/, with its type, size and (if known) test accuracy."""
    out = []
    for path in sorted(CKPT_DIR.glob("*.pt")):
        try:
            out.append(describe(path))
        except Exception as e:  # unreadable or foreign file: list it as unusable rather than hiding it
            out.append({"name": path.name, "kind": None, "label": path.stem, "error": str(e)})
    return {"models": out, "active": state["name"]}


@app.post("/api/model")
def select_model(name: str):
    try:
        load_model(checkpoint_path(name))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return JSONResponse({"error": f"cannot load {name}: {e}"}, status_code=400)
    return status()


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

    model, kind, encoder = state["model"], state["kind"], state["encoder"]
    # centre window first, then +-80 ms shifted copies (test-time augmentation)
    windows = [window_at(waveform, best + d) for d in (0, -SHIFT_SAMPLES, SHIFT_SAMPLES)]
    mfccs = torch.stack([encoder.waveform_to_mfcc(w) for w in windows])  # (3, T, n_mfcc) in [0, 1]

    with torch.no_grad():
        if kind == "snn":
            spikes_in = torch.bernoulli(mfccs.repeat_interleave(state["n_samples"], dim=0))
            logits, hidden_spikes, readout_trace = model(spikes_in, return_spikes=True)
        else:
            logits = model(mfccs)
    logits = logits.mean(dim=0)
    probs_all = torch.softmax(logits, dim=0)
    probs = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=0)
    pred = int(probs.argmax())

    result = {
        "word": KEYWORDS[pred],
        "confidence": float(probs[pred]),
        "probs": {w: float(p) for w, p in zip(KEYWORDS, probs)},
        "probs_unmasked": {w: float(p) for w, p in zip(KEYWORDS, probs_all)},
        "kind": kind,
        "model": state["name"],
        "model_loaded": state["name"] is not None,
    }
    if kind != "snn":
        result.update(raster=None, readout_trace=None,
                      stats={"macs": dense_macs(model, kind), "latency_ms": (time.perf_counter() - t0) * 1000})
        return result

    # --- activity / efficiency stats, for one spike sample (what a chip would actually run) ---
    in_spk = spikes_in[0]                    # (T, 20)
    h_spk = [h[0] for h in hidden_spikes]    # [(T, 128), (T, 128)]
    T = in_spk.shape[0]
    sizes = [layer.synapse.out_features for layer in model.hidden_layers]
    rec = [layer.recurrent is not None for layer in model.hidden_layers]
    n_out = model.readout.synapse.out_features
    # fan-out of each spike source: next layer, plus its own layer when recurrent
    fan_outs = [sizes[0]] + [(s if r else 0) + nxt for s, r, nxt in zip(sizes, rec, sizes[1:] + [n_out])]
    layer_inputs = [in_spk] + h_spk
    # event-driven cost: each spike triggers one synaptic accumulate per outgoing connection
    sops = int(sum(s.sum().item() * fo for s, fo in zip(layer_inputs, fan_outs)))
    # dense (CNN/LSTM-style) cost: every input multiplies every weight at every step
    dense_macs_eq = int(sum(s.shape[1] * fo * T for s, fo in zip(layer_inputs, fan_outs)))
    hidden_total = sum(int(h.sum()) for h in h_spk)
    hidden_capacity = sum(h.numel() for h in h_spk)

    def raster(s):
        t, n = torch.nonzero(s, as_tuple=True)
        return [t.tolist(), n.tolist()]

    result.update(
        raster={"input": raster(in_spk), "hidden": [raster(h) for h in h_spk],
                "shape": {"T": T, "input": in_spk.shape[1], "hidden": [h.shape[1] for h in h_spk]}},
        readout_trace=readout_trace.mean(dim=0).tolist(),  # (T, 10)
        stats={"hidden_spikes": hidden_total, "sparsity": 1 - hidden_total / hidden_capacity, "sops": sops,
               "dense_macs": dense_macs_eq, "latency_ms": (time.perf_counter() - t0) * 1000},
    )
    return result


# ---------------- voice calibration ----------------

@app.post("/api/calibrate/start")
def calibrate_start():
    session = RECORDINGS_DIR / time.strftime("session_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    state["session"] = session
    return {"session": session.name, "words": COMMAND_WORDS}


def calibration_base():
    """The model calibration starts from: the active one, or the base of an active personal model."""
    base = meta(CKPT_DIR / state["name"]).get("base") if state["name"] else None
    if base and (CKPT_DIR / base).is_file():
        model, kind = build_model(torch.load(CKPT_DIR / base, map_location="cpu", weights_only=True))
        return model.eval(), kind, base
    return state["model"], state["kind"], state["name"]


@app.post("/api/calibrate/clip")
async def calibrate_clip(request: Request):
    """Save one recorded clip for `word`; report what the base model thinks it is."""
    word = request.query_params.get("word")
    if word not in COMMAND_WORDS or state["session"] is None:
        return JSONResponse({"error": "call /api/calibrate/start first, with a valid ?word="}, status_code=400)
    clip = save_clip(state["session"], word, read_pcm(await request.body()))
    saved = len(list(state["session"].glob(f"{word}_*.wav")))
    model, kind, _ = calibration_base()
    logits = logits_for(model, kind, state["encoder"].waveform_to_mfcc(clip).unsqueeze(0))
    heard = KEYWORDS[int(logits.masked_fill(~mask_for(COMMAND_WORDS), float("-inf")).argmax())]
    return {"saved": saved, "base_heard": heard, "base_correct": heard == word}


@app.post("/api/correct")
async def correct(request: Request):
    """Save the last utterance under the word the user says it really was."""
    word = request.query_params.get("word")
    if word not in COMMAND_WORDS:
        return JSONResponse({"error": "invalid ?word="}, status_code=400)
    save_clip(CORRECTIONS_DIR, word, read_pcm(await request.body()))
    return {"corrections": n_corrections()}


@app.post("/api/calibrate/train")
def calibrate_train():
    if not train_lock.acquire(blocking=False):
        return JSONResponse({"error": "already training"}, status_code=409)
    try:
        if state["name"] is None:
            return JSONResponse({"error": "load a trained model first"}, status_code=400)
        clips = []
        for f in sorted(RECORDINGS_DIR.glob("*/*.wav")):  # every session + corrections
            audio, _ = sf.read(f, dtype="float32")
            clips.append((torch.from_numpy(audio), f.stem.rsplit("_", 1)[0]))
        if len({w for _, w in clips}) < len(COMMAND_WORDS):
            return JSONResponse({"error": "record at least one clip per command"}, status_code=400)
        base_model, kind, base_name = calibration_base()
        model, report = calibrate(base_model, clips, state["encoder"], mask_for(COMMAND_WORDS))
        out = CKPT_DIR / f"{Path(base_name).stem}_personal.pt"
        torch.save(model.state_dict(), out)
        out.with_suffix(".json").write_text(json.dumps({
            "label": f"{describe(CKPT_DIR / base_name)['label']}, calibrated to your voice",
            "base": base_name,
            "note": f"fine-tuned on {report['n_clips']} of your recordings",
        }, indent=2))
        load_model(out)
        return {**report, "model": out.name}
    finally:
        train_lock.release()


@app.post("/api/calibrate/reset")
def calibrate_reset():
    """Delete the active personal model and go back to the model it was calibrated from."""
    base = meta(CKPT_DIR / state["name"]).get("base") if state["name"] else None
    if not base:
        return JSONResponse({"error": "the active model is not a calibrated one"}, status_code=400)
    personal = CKPT_DIR / state["name"]
    load_model(checkpoint_path(base))
    personal.unlink(missing_ok=True)
    personal.with_suffix(".json").unlink(missing_ok=True)
    return status()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="file in checkpoints/ to start with (any can be chosen later in the UI)")
    parser.add_argument("--samples", type=int, default=8, help="Bernoulli spike samples averaged per window (SNN)")
    parser.add_argument("--port", type=int, default=8737)
    parser.add_argument("--base_only", action="store_true", help="start with the base model even if a calibrated one exists")
    args = parser.parse_args()
    state["n_samples"] = args.samples

    initialize_model(args.checkpoint, args.base_only)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
