# NeuroChair — voice-controlled wheelchair

A 3D web app that drives a wheelchair using a from-scratch **Spiking Neural Network** keyword spotter
(model + training: [snn-keyword-spotter](https://github.com/rushabu/snn-keyword-spotter)).

Browser mic → voice-activity detection → FastAPI server → MFCC → spike train → SNN → command → 3D wheelchair.

| Say | Action |
|-----|--------|
| `go` / `stop` | drive / brake |
| `left` / `right` | turn 90° |
| `up` / `down` | speed gear + / − (toggleable) |
| `on` / `off` | headlights |

`yes` / `no` are masked out, so the SNN only chooses among the commands the chair listens for.
Low-confidence predictions are ignored, `stop` has a lower confidence gate as a safety bias, and the
chair auto-brakes before obstacles.

## Setup

1. Put the trained checkpoint at `checkpoints/snn_checkpoint.pt`. It comes from the Kaggle notebook: the SNN training cell
   saves it to `outputs/snn_checkpoint.pt`.
2. Install and run:
   ```bash
   pip install -r requirements.txt
   python server.py
   ```
3. Open http://localhost:8000 in Chrome or Edge. Mic access needs `localhost` or HTTPS.

Options: `--checkpoint <path>`, `--hidden_sizes 128 128` (must match training), `--samples 8`, `--port 8000`.
Without a checkpoint it runs an untrained model and the UI shows an "untrained model" warning.

Keyboard: W go · S/Space stop · A/D turn · Q/E speed · L lights · C orbit camera · R reset · M mic.

## Calibrate to your voice (recommended)

The base SNN reaches 58.6% on the Speech Commands test set, and a live laptop mic differs from that
data (mic, room, accent). Click **Calibrate to my voice**, say each of the 8 commands 5 times, and the
SNN is fine-tuned on your recordings (about 30 s on CPU). The UI then shows the original vs calibrated accuracy
on takes held out from fine-tuning.

- Recordings are saved in `recordings/session_*/`; the calibrated weights in `checkpoints/snn_personal.pt`,
  which load automatically on the next start (`--base_only` ignores them).
- **Reset to base model** in the UI deletes `snn_personal.pt` and goes back to the original weights.
- Calibration is per speaker: it improves accuracy for the person who recorded and may reduce it for others.

## Project layout

```
server.py              FastAPI backend: audio -> spikes -> SNN -> prediction + spike rasters
personalize.py         voice calibration: augmentation + fine-tuning the SNN on the user's clips
snn/                   LIF neuron, SNN layers, classifier (copied from snn-keyword-spotter)
data/speech_commands.py  MFCC -> spike encoder, same as training (copied)
static/                3D frontend (Three.js, no build step)
  js/scene.js          wheelchair + world, lights, physics, effects
  js/audio.js          mic capture + voice activity detection
  js/app.js            HUD, decoder panel, command logic
checkpoints/           put snn_checkpoint.pt here
```

`snn/` and `data/` must stay identical to the versions the checkpoint was trained with.

## Decoder panel

- **Probabilities**: softmax over the active commands, averaged over 8 Bernoulli spike samples × 3 time shifts (0, ±80 ms)
- **Spike raster**: input spikes (20 MFCC channels) and both LIF hidden layers over 50 time steps
- **Silent neurons**: fraction of hidden neuron-timesteps that did not spike
- **Synaptic ops vs dense**: event-driven accumulates (spikes × fan-out) vs the MACs a dense net of the same shape would do
