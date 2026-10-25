# NeuroChair — voice-controlled wheelchair

A 3D web app that drives a wheelchair by voice. It runs the keyword spotters from
[snn-keyword-spotter](https://github.com/rushabu/snn-keyword-spotter): a from-scratch **spiking neural
network (SNN)** by default, and the non-spiking **CNN** and **LSTM** baselines, selectable in the UI.

Browser mic → voice-activity detection → FastAPI server → MFCC → (spike train →) model → command → 3D wheelchair.

| Say | Action |
|-----|--------|
| `go` / `stop` | drive / brake |
| `left` / `right` | turn 90° |
| `up` / `down` | speed gear + / − (toggleable) |
| `on` / `off` | headlights |

`yes` / `no` are masked out, so the model only chooses among the commands the chair listens for.
Low-confidence predictions are ignored, `stop` has a lower confidence gate as a safety bias, and the
chair auto-brakes before obstacles.

## Setup

1. Put one or more trained checkpoints in `checkpoints/` (see [Models](#models)). `train.py` in
   snn-keyword-spotter writes `outputs/snn_checkpoint.pt`, `outputs/cnn_checkpoint.pt` and
   `outputs/lstm_checkpoint.pt`.
2. Install and run:
   ```bash
   pip install -r requirements.txt
   python server.py
   ```
3. Open http://localhost:8737 in Chrome or Edge. Mic access needs `localhost` or HTTPS.

Options: `--checkpoint <file in checkpoints/>` (start model, default `snn_checkpoint.pt`), `--samples 8`
(spike samples per window, SNN only), `--port 8737`, `--base_only` (ignore a calibrated model at start).
Without a loadable checkpoint it runs an untrained SNN and the UI shows an "untrained model" warning.

Keyboard: W go · S/Space stop · A/D turn · Q/E speed · L lights · C orbit camera · R reset · M mic.

## Models

**Settings → Model** lists every `.pt` file in `checkpoints/` and switches between them instantly, with
no restart. The server works out from the weights whether a file is an SNN (any layer sizes, with or
without recurrence), a CNN or an LSTM, so a new checkpoint only has to be copied into the folder.

An optional sidecar `<name>.json` next to a checkpoint sets what the UI shows:

```json
{ "label": "SNN, recurrent 2x128 (paper model)", "test_accuracy": 0.728, "note": "50 epochs, full v0.02 split." }
```

| Kind | What the UI shows |
|------|-------------------|
| SNN | spike raster, hidden spikes, silent neurons, synaptic ops vs dense; 8 spike samples × 3 time shifts averaged |
| CNN / LSTM | no raster (they don't spike); multiply-adds per pass; 3 time shifts averaged |

## Calibrate to your voice (recommended)

A live laptop mic differs from the Speech Commands recordings (mic, room, accent). Click
**Calibrate to my voice**, say each of the 8 commands 5 times, and the **active model** is fine-tuned on
your recordings (seconds for the CNN/LSTM, about a minute for the SNN on CPU). The UI then shows the
original vs calibrated accuracy on takes held out from fine-tuning.

- Recordings are saved in `recordings/session_*/`; the calibrated weights in
  `checkpoints/<model>_personal.pt`, which appears in the model list. `snn_checkpoint_personal.pt` is
  picked automatically at start (`--base_only` ignores it).
- **Back to …** in the UI deletes the active calibrated model and returns to the one it came from.
- Calibration is per speaker: it improves accuracy for the person who recorded and may reduce it for others.

## Project layout

```
server.py              FastAPI backend: model registry, audio -> MFCC (-> spikes) -> prediction
personalize.py         voice calibration: augmentation + fine-tuning the active model on the user's clips
snn/                   LIF neuron, SNN layers, classifier, CNN/LSTM baselines (copied from snn-keyword-spotter)
data/speech_commands.py  MFCC -> spike encoder, same as training (copied)
static/                3D frontend (Three.js, no build step)
  js/scene.js          wheelchair + world, lights, physics, effects
  js/audio.js          mic capture + voice activity detection
  js/app.js            HUD, decoder panel, model picker, command logic
checkpoints/           *.pt weights (+ optional *.json labels); every file is selectable in the UI
```

`snn/` and `data/` must stay identical to the versions the checkpoints were trained with.

## Decoder panel

- **Probabilities**: softmax over the active commands, averaged over 3 time shifts (0, ±80 ms) and, for the SNN, 8 Bernoulli spike samples each
- **Spike raster** (SNN): input spikes (20 MFCC channels) and both LIF hidden layers over 50 time steps
- **Silent neurons** (SNN): fraction of hidden neuron-timesteps that did not spike
- **Synaptic ops vs dense** (SNN): event-driven accumulates (spikes × fan-out) vs the MACs a dense net of the same shape would do
- **Multiply-adds** (CNN/LSTM): arithmetic of one forward pass
