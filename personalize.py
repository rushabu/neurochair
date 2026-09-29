"""
Per-user voice calibration.

The base SNN was trained on Google Speech Commands, which is recorded on different
microphones, rooms and voices than a live laptop mic. With only 58% accuracy on its own
test set, that domain shift hurts a lot live. Calibration records a few examples of each
command from the actual user and fine-tunes the trained SNN on them (starting from the
trained weights, not from scratch), with augmentation so a handful of clips go a long way:
  - random time shift (+-100 ms) -- the word won't always sit in the same place
  - random gain                  -- distance to the mic varies
  - a little additive noise      -- background changes
  - fresh Bernoulli spike samples every step (rate coding is stochastic anyway)
"""

import copy
import random
import time

import torch
import torch.nn.functional as F

from data.speech_commands import KEYWORDS, SAMPLE_RATE, LABEL_TO_IDX


def augment(w: torch.Tensor) -> torch.Tensor:
    w = w.clone()
    shift = random.randint(-SAMPLE_RATE // 10, SAMPLE_RATE // 10)
    w = torch.roll(w, shift)
    if shift > 0:
        w[:shift] = 0
    elif shift < 0:
        w[shift:] = 0
    w = w * random.uniform(0.6, 1.4)
    peak = w.abs().max().clamp(min=1e-4)
    return w + torch.randn_like(w) * peak * random.uniform(0.0, 0.03)


def build_features(clips, encoder, n_aug: int):
    """clips: list of (waveform, word). Returns (mfccs (N, T, F), labels (N,))."""
    feats, labels = [], []
    for w, word in clips:
        for k in range(n_aug + 1):
            feats.append(encoder.waveform_to_mfcc(w if k == 0 else augment(w)))
            labels.append(LABEL_TO_IDX[word])
    return torch.stack(feats), torch.tensor(labels)


@torch.no_grad()
def accuracy(model, clips, encoder, mask: torch.Tensor, n_samples: int = 8) -> float:
    if not clips:
        return float("nan")
    model.eval()
    correct = 0
    for w, word in clips:
        mfcc = encoder.waveform_to_mfcc(w)
        x = torch.bernoulli(mfcc.unsqueeze(0).expand(n_samples, -1, -1).contiguous())
        logits = model(x).mean(0).masked_fill(~mask, float("-inf"))
        correct += int(KEYWORDS[int(logits.argmax())] == word)
    return correct / len(clips)


def fine_tune(base_model, clips, encoder, steps: int = 150, lr: float = 1e-3, batch: int = 32, n_aug: int = 30):
    # 150 steps matched 300 on held-out takes in testing. The SNN's per-timestep Python loop
    # runs on tiny tensors, where one CPU thread is ~3x faster than many (less sync overhead).
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = copy.deepcopy(base_model)
        X, y = build_features(clips, encoder, n_aug)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(steps):
            idx = torch.randint(0, len(X), (batch,))
            logits = model(torch.bernoulli(X[idx]))
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        return model
    finally:
        torch.set_num_threads(threads)


def split_holdout(clips, per_word: int):
    by_word = {}
    for c in clips:
        by_word.setdefault(c[1], []).append(c)
    train, test = [], []
    for word, items in by_word.items():
        random.shuffle(items)
        k = per_word if len(items) > per_word + 1 else 0
        test += items[:k]
        train += items[k:]
    return train, test


def calibrate(base_model, clips, encoder, mask):
    """
    1. Score the base model on ALL the user's clips (it has never seen them).
    2. Fine-tune on some clips, score on held-out clips -> honest personalised accuracy.
    3. Fine-tune again on everything -> the model actually used from now on.
    """
    t0 = time.perf_counter()
    base_acc = accuracy(base_model, clips, encoder, mask)
    train, test = split_holdout(clips, per_word=2)
    heldout_base = accuracy(base_model, test, encoder, mask)
    heldout_personal = accuracy(fine_tune(base_model, train, encoder), test, encoder, mask) if test else float("nan")
    final = fine_tune(base_model, clips, encoder)
    return final, {
        "n_clips": len(clips),
        "base_acc_all": base_acc,
        "heldout_n": len(test),
        "heldout_base_acc": heldout_base,
        "heldout_personal_acc": heldout_personal,
        "seconds": time.perf_counter() - t0,
    }
