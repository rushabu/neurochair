"""
Noise injection for the noise-robustness use case (Phase 5, Step 8).

Applied to the raw waveform BEFORE MFCC/spike encoding, so it degrades the
signal the same way real-world noise would, rather than corrupting spikes
directly.

  - white: additive Gaussian noise
  - babble: sum of several shifted copies of the same waveform (crude
    multi-talker approximation, no external audio needed)
  - street: low-pass-filtered white noise (approximates the rumble-heavy
    spectrum of street/traffic noise) mixed with white noise
"""

import torch
import torchaudio


def _snr_scale(signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    sig_power = signal.pow(2).mean()
    noise_power = noise.pow(2).mean().clamp(min=1e-10)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    return noise * (target_noise_power / noise_power).sqrt()


def add_white_noise(waveform: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
    noise = torch.randn_like(waveform)
    return waveform + _snr_scale(waveform, noise, snr_db)


def add_babble_noise(waveform: torch.Tensor, snr_db: float = 10.0, n_voices: int = 4) -> torch.Tensor:
    babble = torch.zeros_like(waveform)
    for _ in range(n_voices):
        shift = torch.randint(0, waveform.shape[-1], (1,)).item()
        babble += torch.roll(waveform, shifts=shift, dims=-1)
    babble = babble / n_voices
    return waveform + _snr_scale(waveform, babble, snr_db)


def add_street_noise(waveform: torch.Tensor, snr_db: float = 10.0, sample_rate: int = 16000) -> torch.Tensor:
    noise = torch.randn_like(waveform)
    # crude low-pass to bias energy toward low frequencies (traffic-like rumble)
    noise = torchaudio.functional.lowpass_biquad(noise, sample_rate, cutoff_freq=400.0)
    return waveform + _snr_scale(waveform, noise, snr_db)


NOISE_FNS = {
    "white": add_white_noise,
    "babble": add_babble_noise,
    "street": add_street_noise,
}


def augment_waveform(waveform: torch.Tensor, max_shift: int = 1600) -> torch.Tensor:
    """Training augmentation: random time shift (up to +-100 ms) and, 80% of the time, noise at 5-30 dB SNR."""
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)))
    w = torch.roll(waveform, shift)
    if shift > 0:
        w[:shift] = 0
    elif shift < 0:
        w[shift:] = 0
    if torch.rand(1).item() < 0.8:
        noise_fn = add_white_noise if torch.rand(1).item() < 0.5 else add_street_noise
        w = noise_fn(w, snr_db=float(torch.empty(1).uniform_(5.0, 30.0)))
    return w
