"""Train and save the CNN/LSTM weights used by the model picker."""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.speech_commands import KEYWORDS, N_MFCC, N_TIME_STEPS, SpeechCommandsSpikes
from snn.baselines import CNNBaseline, LSTMBaseline

ROOT = Path(__file__).resolve().parent


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        total += len(y)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="folder containing word directories and official validation/testing lists")
    parser.add_argument("--download", action="store_true", help="download Speech Commands v0.02 (multiple GB)")
    parser.add_argument("--epochs", type=positive, default=50)
    parser.add_argument("--batch-size", type=positive, default=64)
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--max-per-class", type=positive, help="optional cap per split for a short trial")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()
    targets = [args.output_dir / f"{kind}_checkpoint.pt" for kind in ("cnn", "lstm")]
    if any(path.exists() for path in targets):
        parser.error("weights already exist; use a different --output-dir to preserve them")
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    root = args.data_root.resolve()
    if args.download:
        from torchaudio.datasets import SPEECHCOMMANDS
        root.mkdir(parents=True, exist_ok=True)
        # Resolve the extracted folder explicitly; avoid backend-dependent audio loading.
        root = Path(SPEECHCOMMANDS(str(root), download=True)._path)
    for name in ("validation_list.txt", "testing_list.txt"):
        if not (root / name).is_file():
            parser.error(f"missing {root / name}; use the official dataset or --download")
    loaders = {}
    for split in ("training", "validation", "testing"):
        ds = SpeechCommandsSpikes(str(root), split=split, output="mfcc",
                                  download_if_missing=False, max_per_class=args.max_per_class,
                                  augment=split == "training")
        if {label for _, label in ds.samples} != set(range(len(KEYWORDS))):
            parser.error(f"{split} must contain audio for all {len(KEYWORDS)} keywords")
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=split == "training")
        print(f"{split}: {len(ds)} clips", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind, model in (("cnn", CNNBaseline(N_MFCC, N_TIME_STEPS, len(KEYWORDS))),
                        ("lstm", LSTMBaseline(N_MFCC, len(KEYWORDS)))):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_score, best_weights = -1, None
        for epoch in range(args.epochs):
            model.train()
            for x, y in loaders["training"]:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
                loss.backward()
                optimizer.step()
            score = accuracy(model, loaders["validation"], device)
            if score > best_score:
                best_score, best_weights = score, copy.deepcopy(model.state_dict())
            print(f"{kind} epoch {epoch + 1}/{args.epochs}: validation={score:.4f}", flush=True)
        model.load_state_dict(best_weights)
        score = accuracy(model, loaders["testing"], device)
        path = args.output_dir / f"{kind}_checkpoint.pt"
        torch.save({key: value.cpu() for key, value in model.state_dict().items()}, path)
        path.with_suffix(".json").write_text(json.dumps({
            "label": f"{kind.upper()} baseline",
            "test_accuracy": score,
            "note": f"Locally trained, {args.epochs} epochs; best validation={best_score:.4f}; "
                    f"{len(loaders['training'].dataset)} training clips, "
                    f"{len(loaders['testing'].dataset)} held-out test clips.",
        }, indent=2), encoding="utf-8")
        print(f"Saved {path}; test accuracy={score:.4f}", flush=True)


if __name__ == "__main__":
    main()
