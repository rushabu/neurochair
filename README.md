# NeuroChair

**A voice-controlled wheelchair simulator powered by selectable SNN, CNN, and LSTM keyword spotters.**

Speak a command into your microphone to drive a wheelchair through a 3D courtyard. NeuroChair runs speech inference locally, displays neural spike activity, and can fine-tune the model on your voice. You can also drive with the keyboard or on-screen buttons.

The model and training pipeline come from [snn-keyword-spotter](https://github.com/rushabu/snn-keyword-spotter).

## Features

- Eight voice commands for driving, steering, speed, and headlights.
- A Three.js scene with a follow camera, orbit camera, minimap, and obstacle braking.
- Local inference through a FastAPI backend with CPU support.
- Switch between SNN, CNN, and LSTM checkpoints in **Settings → Model** without restarting.
- Voice calibration and corrections saved between sessions.
- A decoder display showing spike rasters, confidence, latency, and estimated synaptic operations.
- Adjustable confidence threshold, microphone sensitivity, and speed command recognition.

## Requirements

The current setup has been verified on **Windows 11 with CPython 3.13.6, 64-bit**, using CPU inference.

| Requirement | Details |
| --- | --- |
| Python | CPython 3.13, 64-bit |
| Browser | Chrome or Edge with WebGL 2 support and graphics acceleration |
| Microphone | Needed for voice commands and calibration; optional for keyboard use |
| Internet | Needed to install packages and load the frontend's CDN modules and fonts |

CUDA, FFmpeg, and Node.js are not required for this application. Audio processing and model inference run on your computer; frontend libraries and fonts load from external services.

## Quick start: this PC

If the project environment is already installed, open PowerShell and run:

```powershell
cd C:\Users\Rushabh\Downloads\Projects\neurochair
.\start.cmd
```

Open [http://localhost:8737](http://localhost:8737) in your browser.

Keep the terminal open while using the app. To stop the server, press **Ctrl+C** in that terminal. Run `start.cmd` again whenever you want to restart it; reinstalling dependencies is unnecessary.

## First-time setup

Run these commands from the project folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\start.cmd
```

No virtual environment activation is needed. In your editor, select `.venv\Scripts\python.exe` as the Python interpreter.

### Trained checkpoint

Place the trained model at:

```text
checkpoints/snn_checkpoint.pt
```

The default trained SNN checkpoint is included in Git, so a fresh clone can run it immediately after installing dependencies. The training notebook saves this file as `outputs/snn_checkpoint.pt`. Additional and personalized checkpoints remain ignored by Git.

At startup, the server reads the checkpoint's layer sizes and recurrence settings. If the checkpoint is missing, the app still opens with an **untrained model** warning and random predictions. Use a trained checkpoint for meaningful voice recognition.

## Models

**Settings → Model** lists every `.pt` file in `checkpoints/` and switches between them without a restart. The architecture is inferred from the weights. Git includes the default trained SNN checkpoint, model code, and metadata. Copy trained `cnn_checkpoint.pt`, `lstm_checkpoint.pt`, or additional SNN weights from the training project's outputs into this folder to make them available.

An optional matching `<name>.json` sets the model label, test accuracy, and note shown in the UI. Metadata alone does not install a model.

The SNN displays spike rasters and synaptic operation estimates. CNN and LSTM models consume continuous MFCCs and display multiply-add counts instead of spike activity.

## Using the app

1. Start the server and open the local URL.
2. Select **Turn on microphone** and allow browser microphone access, or select **Use keyboard only**.
3. Say one command at a time, leaving a short pause between words.
4. Open **Settings** to adjust recognition or calibrate the model to your voice.

### Voice commands

| Command | Action |
| --- | --- |
| `go` | Drive forward |
| `stop` | Brake |
| `left` | Turn left by 90 degrees |
| `right` | Turn right by 90 degrees |
| `up` | Increase speed gear |
| `down` | Decrease speed gear |
| `on` | Turn headlights on |
| `off` | Turn headlights off |

There are three speed gears. Recognition of `up` and `down` can be disabled in Settings. The model's `yes` and `no` classes are excluded from wheelchair commands.

Low-confidence predictions are ignored. The `stop` command uses a lower confidence threshold, and the simulated chair automatically brakes before obstacles and the courtyard boundary.

### Keyboard controls

| Key | Action |
| --- | --- |
| W / Up arrow | Drive forward |
| S / Down arrow / Space | Brake |
| A / Left arrow | Turn left |
| D / Right arrow | Turn right |
| E / + | Increase speed gear |
| Q / - | Decrease speed gear |
| L | Toggle headlights |
| M | Toggle microphone |
| C | Switch between follow and orbit cameras |
| R | Reset the chair to its starting position |
| Esc | Close Settings |

You can also click the command buttons. In orbit mode, drag to look around. When a button has focus, Space activates that button.

## Calibrate to your voice

Differences in microphones, accents, and room noise can affect recognition. Calibration fine-tunes a copy of the active model using your recordings.

1. Open **Settings → Calibrate to my voice**.
2. Say each of the eight commands five times as prompted: 40 recordings in total.
3. Wait for training to finish. Runtime depends on your CPU.
4. Review the before/after scores, which use recordings held out from fine-tuning, then select **Start driving**.

The final personalized model is trained on all saved recordings and appears in the model list. `snn_checkpoint_personal.pt` loads automatically on the next start when available. Calibration results depend on the speaker and recordings; an improvement is not guaranteed.

### Correct a misheard command

After a misheard utterance, hold **Shift** and press the key for the intended command. For example, **Shift+W** saves the last utterance as `go` and executes the drive command. Open Settings and choose **Retrain with corrections** to train on the saved clips.

Retraining uses clips from all saved calibration sessions and corrections. For consistent personalization, keep these recordings associated with the same speaker.

### Saved files

| Location | Contents |
| --- | --- |
| `checkpoints/snn_checkpoint.pt` | Base trained model |
| `checkpoints/<model>_personal.pt` | Personalized model |
| `recordings/session_*/` | Calibration recordings |
| `recordings/corrections/` | Corrected utterances |

**Back to …** removes the active personalized checkpoint and switches back to its base model. It keeps your recordings, which can be used during later retraining. To temporarily ignore personalized weights, launch with `--base_only`.

## Launch options

```powershell
# Use a different port, then open http://localhost:8001
.\start.cmd --port 8001

# Use a specific trained checkpoint
.\start.cmd --checkpoint cnn_checkpoint.pt

# Ignore personalized weights for this run
.\start.cmd --base_only

# Average fewer spike samples per audio window
.\start.cmd --samples 4
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--port` | `8737` | Local server port |
| `--checkpoint` | `checkpoints/snn_checkpoint.pt` | Base checkpoint file in `checkpoints/` |
| `--samples` | `8` | Bernoulli spike samples per window; use a positive integer |
| `--base_only` | Disabled | Start with the base model |

The server listens on `127.0.0.1`. The Windows launcher uses the project's virtual environment and resolves the server path relative to its own location.

You can also launch directly from the project folder:

```powershell
.\.venv\Scripts\python.exe server.py
```

Or use Uvicorn with the default model settings:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8737
```

## How recognition works

```text
Browser microphone
  → voice activity detection
  → mono 16 kHz float32 audio
  → loudest one-second window and two shifted copies
  → MFCC features
  → Bernoulli spike trains
  → spiking neural network
  → averaged logits and command masking
  → confidence gate
  → simulated wheelchair action
```

For the SNN, inference uses eight spike samples for each of three windows: the selected window and copies shifted by ±80 ms. The encoder produces 20 MFCC channels over 50 time steps. MFCC extraction uses TorchAudio; WAV reading and writing use SoundFile.

The decoder's spike raster shows input and hidden-layer activity. **Silent neurons** measures the fraction of hidden neuron time steps without spikes. **Synaptic ops** estimates event-driven accumulations against dense multiply-accumulate operations for the same layer shapes; this is an operation estimate, not a measured hardware energy saving.

## Dependencies and maintenance

`requirements.txt` pins the direct Python dependencies. `requirements-lock.txt` pins the resolved dependencies used by the verified Windows environment. Use the lockfile for repeatable installation.

The frontend pins Three.js 0.186.1 in `static/index.html` and has no build step. TorchAudio 2.11 supports PyTorch 2.11 and later; see its [installation guide](https://docs.pytorch.org/audio/master/installation.html) when changing versions.

Preserve the model architecture and preprocessing behavior when making changes to `snn/` or `data/`: the checkpoint depends on them. To roll back a dependency upgrade, restore both requirements files from the previous Git revision and install into a fresh virtual environment. Keep checkpoints and recordings when replacing the environment.

## Project structure

```text
neurochair/
├── server.py                 # FastAPI routes, model loading, inference
├── personalize.py            # Voice augmentation and fine-tuning
├── start.cmd                 # Windows launcher
├── requirements.txt          # Direct Python dependency pins
├── requirements-lock.txt     # Resolved Python dependency pins
├── snn/                      # SNN layers, classifier, CNN/LSTM baselines
├── data/                     # MFCC extraction, spike encoding, audio utilities
├── static/
│   ├── index.html            # Interface and frontend dependency imports
│   ├── styles.css            # Interface styling
│   └── js/
│       ├── app.js            # Controls, decoder, calibration interface
│       ├── audio.js          # Microphone capture and voice activity detection
│       └── scene.js          # 3D courtyard, wheelchair, animation, collisions
├── checkpoints/              # Base and personalized model weights
└── recordings/               # Created when clips are saved
```

## Troubleshooting

| Problem | What to do |
| --- | --- |
| Launcher says the environment is missing | Complete First-time setup from the project folder. |
| `ModuleNotFoundError` | Install `requirements-lock.txt` using `.venv\Scripts\python.exe`, then run `start.cmd`. Check your editor's interpreter too. |
| Python 3.13 is not found | Install 64-bit CPython 3.13 with the Windows Python launcher, then reopen PowerShell. |
| Port 8737 is already in use | Stop the previous server with Ctrl+C, or launch with `--port 8001` and open the matching URL. |
| Browser cannot connect | Keep the server terminal open and check its startup output and port. |
| Microphone does not work | Open the app through `localhost` or `127.0.0.1`, allow browser microphone access, and check the input device and Windows microphone permissions. |
| Scene is blank | Check browser graphics acceleration, WebGL 2 support, and access to the jsDelivr CDN. |
| Model is untrained | Put the trained checkpoint in `checkpoints/snn_checkpoint.pt`, then restart. |
| Commands are misheard | Calibrate in your current room with your current microphone. Adjust sensitivity and confidence in Settings. |
| Calibration does not match the checkpoint | Recalibrate after changing the base model, or start with `--base_only`. |

## Validation of the current environment

The dependency update was checked with checkpoint loading, comparison of MFCC features and model outputs against the previous runtime, prediction response serialization, a full calibration/save/reload/reset cycle using synthetic clips, dependency consistency, live HTTP requests, browser rendering, and driving controls.

Physical microphone capture and live spoken-command accuracy still require a check with your microphone and voice. Synthetic calibration checks establish that the training path runs; they do not measure recognition accuracy.
