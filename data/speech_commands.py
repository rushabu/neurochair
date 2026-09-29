"""
Google Speech Commands -> MFCC -> spike-train pipeline.

Designed to run unchanged in two places:
  1. Kaggle notebook: point DATA_ROOT at a Kaggle "Speech Commands" dataset
     mounted under /kaggle/input/... (folder-per-word .wav layout). No
     internet needed if the dataset is attached to the kernel.
  2. Anywhere else with internet: leave DATA_ROOT pointing at a writable
     folder that doesn't exist yet, and torchaudio will download the
     official v0.02 archive there automatically.

Reduced keyword set (10 core commands, matches the standard TF Speech
Commands "core words" task -- easy to extend to the full 35-word set later
by editing KEYWORDS).
"""

import os
import glob
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset

from .noise import augment_waveform

KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABEL_TO_IDX = {w: i for i, w in enumerate(KEYWORDS)}

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

N_MFCC = 20          # number of MFCC coefficients -> input feature dimension
N_TIME_STEPS = 50    # number of time frames fed to the SNN (also the ablation "time steps" knob)


def find_kaggle_speech_commands_root(explicit_root: str | None = None) -> str | None:
    """
    Try to locate a Speech Commands-style dataset already mounted under
    /kaggle/input (folder-per-word .wav layout). Returns the directory that
    directly contains the word subfolders, or None if not found.
    """
    candidates = []
    if explicit_root:
        candidates.append(explicit_root)
    if os.path.isdir("/kaggle/input"):
        candidates.extend(glob.glob("/kaggle/input/*"))
        candidates.extend(glob.glob("/kaggle/input/*/*"))

    for cand in candidates:
        if not os.path.isdir(cand):
            continue
        # Heuristic: a valid root has at least a few of our keyword folders
        # directly inside it, each containing .wav files.
        hits = sum(1 for kw in KEYWORDS if os.path.isdir(os.path.join(cand, kw)))
        if hits >= 5:
            return cand
    return None


class MFCCSpikeEncoder:
    """
    Waveform -> MFCC -> normalized -> spike train.

    coding='rate': each (time, feature) MFCC value in [0, 1] is treated as a
        spike probability; a Bernoulli draw is taken at every time step
        (independently), so higher-energy frames spike more often.
    coding='temporal': each feature emits exactly one spike per clip, at a
        latency inversely proportional to its magnitude (stronger signal ->
        earlier spike). This is standard "latency coding" and is used for
        the coding-scheme ablation.
    """

    def __init__(self, coding: str = "rate", n_mfcc: int = N_MFCC, n_time_steps: int = N_TIME_STEPS):
        assert coding in ("rate", "temporal")
        self.coding = coding
        self.n_time_steps = n_time_steps
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": CLIP_SAMPLES // n_time_steps, "n_mels": 40},
        )

    def waveform_to_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (samples,) -> mfcc: (time_steps, n_mfcc), normalized to [0, 1] PER FEATURE."""
        mfcc = self.mfcc_transform(waveform.unsqueeze(0)).squeeze(0)  # (n_mfcc, time)
        mfcc = mfcc.transpose(0, 1)  # (time, n_mfcc)

        # Pad/crop to a fixed number of time steps
        t = mfcc.shape[0]
        if t < self.n_time_steps:
            mfcc = F.pad(mfcc, (0, 0, 0, self.n_time_steps - t))
        else:
            mfcc = mfcc[: self.n_time_steps]

        # Per-FEATURE min-max normalization -- each of the n_mfcc coefficients is scaled
        # independently across the time axis, NOT a single global min/max over the whole
        # (time, n_mfcc) block. This matters a lot in practice: MFCC coefficient 0 (overall
        # log-energy) typically has a much larger dynamic range than the higher-order
        # coefficients, which carry most of the actual phonetic/spectral-shape information
        # that distinguishes one spoken word from another -- especially on real 1-second
        # clips that are mostly silence outside the spoken word, which pushes c0's range even
        # higher. A single global min/max lets c0 dominate the [0,1] scale and crushes the
        # other ~19 coefficients into a narrow sub-band, destroying most of their
        # discriminative variance before it ever reaches the model -- for the SNN (via
        # near-constant Bernoulli firing rates) AND for the CNN/LSTM baselines (via starved
        # input variance), which is why both were previously failing near chance together.
        feat_min = mfcc.min(dim=0, keepdim=True).values  # (1, n_mfcc)
        feat_max = mfcc.max(dim=0, keepdim=True).values  # (1, n_mfcc)
        denom = (feat_max - feat_min).clamp(min=1e-6)
        mfcc = (mfcc - feat_min) / denom
        return mfcc

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        mfcc = self.waveform_to_mfcc(waveform)  # (time, features), in [0, 1]

        if self.coding == "rate":
            spikes = torch.bernoulli(mfcc)  # independent draw per (time, feature)
            return spikes

        # temporal / latency coding: one spike per feature, timed by magnitude
        n_time, n_feat = mfcc.shape
        # stronger magnitude -> smaller latency (fires earlier)
        latency = ((1.0 - mfcc.max(dim=0).values) * (n_time - 1)).round().long()  # (n_feat,)
        spikes = torch.zeros(n_time, n_feat)
        spikes[latency, torch.arange(n_feat)] = 1.0
        return spikes


class SpeechCommandsSpikes(Dataset):
    """
    Returns (spike_train, label_idx) pairs.
      spike_train: (time_steps, n_mfcc) float tensor of 0/1 spikes
      label_idx:   int in [0, len(KEYWORDS))
    """

    def __init__(self, data_root: str, split: str = "training",
                 coding: str = "rate", n_mfcc: int = N_MFCC, n_time_steps: int = N_TIME_STEPS,
                 max_per_class: int | None = None, download_if_missing: bool = True,
                 output: str = "spikes", noise_fn=None, augment: bool = False):
        """
        output: 'spikes' (default, for the SNN) or 'mfcc' (continuous, for
        the CNN/LSTM baselines). Both share the exact same file list, split,
        and MFCC frontend -- only the final encoding step differs -- so the
        SNN and baselines are trained/evaluated on identical audio.

        noise_fn: optional callable(waveform) -> waveform applied BEFORE MFCC
        extraction, for the noise-robustness eval (see data/noise.py). Leave
        None for clean audio.
        """
        assert output in ("spikes", "mfcc")
        self.output = output
        self.noise_fn = noise_fn
        self.augment = augment
        self.encoder = MFCCSpikeEncoder(coding=coding, n_mfcc=n_mfcc, n_time_steps=n_time_steps)
        self.samples = []  # list of (filepath, label_idx)

        kaggle_root = find_kaggle_speech_commands_root(data_root)
        if kaggle_root is not None:
            self._load_from_folder(kaggle_root, split, max_per_class)
        elif download_if_missing:
            self._load_via_torchaudio(data_root, split, max_per_class)
        else:
            raise FileNotFoundError(
                f"No Speech Commands data found at '{data_root}' or under /kaggle/input, "
                "and download_if_missing=False."
            )

    def _load_from_folder(self, root: str, split: str, max_per_class):
        # Standard layout: <root>/<word>/<file>.wav, with optional
        # testing_list.txt / validation_list.txt for the official split.
        split_files = self._read_official_split(root, split)
        val_files = self._read_official_split(root, "validation")
        test_files = self._read_official_split(root, "testing")
        has_official_lists = val_files is not None or test_files is not None

        for word in KEYWORDS:
            word_dir = os.path.join(root, word)
            if not os.path.isdir(word_dir):
                continue
            wavs = sorted(glob.glob(os.path.join(word_dir, "*.wav")))

            if split == "training":
                if has_official_lists:
                    # training = everything NOT listed in the official val/test files
                    excluded = (val_files or set()) | (test_files or set())
                    wavs = [w for w in wavs
                            if f"{word}/{os.path.basename(w)}" not in excluded]
                else:
                    wavs = [w for w in wavs if self._hash_split(w) == "training"]
            elif split_files is not None:
                wavs = [w for w in wavs if f"{word}/{os.path.basename(w)}" in split_files]
            else:
                # No official split file available -> deterministic hash split
                wavs = [w for w in wavs if self._hash_split(w) == split]

            if max_per_class:
                wavs = wavs[:max_per_class]
            self.samples.extend((w, LABEL_TO_IDX[word]) for w in wavs)

    @staticmethod
    def _read_official_split(root: str, split: str):
        fname = {"validation": "validation_list.txt", "testing": "testing_list.txt"}.get(split)
        if fname is None:
            return None  # "training" = everything not in val/test, handled via hash fallback for simplicity
        path = os.path.join(root, fname)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return set(line.strip() for line in f)

    @staticmethod
    def _hash_split(path: str, val_frac: float = 0.1, test_frac: float = 0.1) -> str:
        """Deterministic split by filename hash, used when no official list is present."""
        h = abs(hash(os.path.basename(path))) % 1000 / 1000.0
        if h < test_frac:
            return "testing"
        if h < test_frac + val_frac:
            return "validation"
        return "training"

    def _load_via_torchaudio(self, data_root: str, split: str, max_per_class):
        subset_map = {"training": "training", "validation": "validation", "testing": "testing"}
        ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=data_root, download=True, subset=subset_map[split]
        )
        counts = {w: 0 for w in KEYWORDS}
        for i in range(len(ds)):
            _, sr, label, *_ = ds.get_metadata(i)
            if label not in LABEL_TO_IDX:
                continue
            if max_per_class and counts[label] >= max_per_class:
                continue
            counts[label] += 1
            fileid = ds._walker[i]
            filepath = os.path.join(ds._path, label, fileid)
            self.samples.append((filepath, LABEL_TO_IDX[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        # Use soundfile directly for loading (avoids relying on torchaudio's
        # load() backend/torchcodec dependency, which varies across
        # environments -- soundfile is a lightweight, reliable constant).
        raw, sr = sf.read(filepath, dtype="float32")
        waveform = torch.from_numpy(raw)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)  # mono

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        # Pad/crop to exactly 1 second
        if waveform.shape[0] < CLIP_SAMPLES:
            waveform = F.pad(waveform, (0, CLIP_SAMPLES - waveform.shape[0]))
        else:
            waveform = waveform[:CLIP_SAMPLES]

        if self.augment:
            waveform = augment_waveform(waveform)
        if self.noise_fn is not None:
            waveform = self.noise_fn(waveform)

        if self.output == "mfcc":
            return self.encoder.waveform_to_mfcc(waveform), label
        spikes = self.encoder.encode(waveform)
        return spikes, label
