"""Layer-wise phone probing on SpeechTokenizer RVQ embeddings.

Reads the cached RVQ features (``features.npz``, one 1024-d vector per 20 ms
frame) and the MFA phone labels (``phones.npz``), then trains a linear probe per
RVQ depth to predict the ARPAbet phone of each frame.

Clips that MFA could not align are skipped automatically. Silence and pause
frames carry label ``-1`` and are excluded from both training and evaluation.

Protocol:
  * split by speaker, so the probe is tested on voices it never saw;
  * standardise features on training frames only;
  * report accuracy excluding silence, against the majority-class baseline and
    per-class recall, because phone frequency is very skewed (AH alone is ~7%).

    .venv/bin/python phone_probe.py
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import train
from config import TrainConfig

DATA_DIR = Path(__file__).parent / "data" / "libritts_preprocessed" / "dev.clean_36speakers_20utts"


def load_frames(data_dir):
    """Collect every labelled frame from clips that MFA aligned.

    Features come from ``frames.npz`` (one vector per frame) or, for the
    back-compatibility path, from ``features.npz``. Returns the padded frame
    tensor, and for each labelled frame its clip row and frame column.
    """
    features_dir = data_dir / "features"
    frame_path = features_dir / "frames.npz"
    if not frame_path.exists():
        raise SystemExit(
            f"{frame_path} not found. A frame-level phone probe needs per-frame "
            "features; run:\n  python speechtokenizer_features.py --dataset-dir "
            f"{data_dir} --frames"
        )
    bundle = np.load(frame_path, allow_pickle=False)
    phones = np.load(features_dir / "phones.npz", allow_pickle=False)

    feature_ids = [str(u) for u in bundle["utterance_ids"]]
    row_of = {utt: i for i, utt in enumerate(feature_ids)}
    grid_ids = [str(u) for u in phones["utterance_ids"]]
    labels, grid_lengths = phones["labels"], phones["lengths"]
    frame_lengths = bundle["lengths"]

    rows, cols, ids = [], [], []
    skipped = 0
    for i, utt in enumerate(grid_ids):
        if utt not in row_of:              # aligned, but no cached features
            skipped += 1
            continue
        r = row_of[utt]
        # Labels and features are both 25 fps, but clip lengths can differ by a
        # frame; trust whichever is shorter.
        n = min(int(grid_lengths[i]), int(frame_lengths[r]))
        keep = np.nonzero(labels[i, :n] >= 0)[0]   # drop silence / pause
        if keep.size == 0:
            skipped += 1
            continue
        rows.append(np.full(keep.size, r, dtype=np.int64))
        cols.append(keep.astype(np.int64))
        ids.append(np.full(keep.size, i, dtype=np.int64))

    rows, cols, ids = np.concatenate(rows), np.concatenate(cols), np.concatenate(ids)
    frame_labels = labels[ids, cols].astype(np.int64)
    speaker = bundle["speaker_ids"][rows]
    phone_names = [str(p) for p in phones["phone_names"]]
    meta = {"frames": bundle["frames"], "lengths": frame_lengths}
    return meta, rows, cols, frame_labels, speaker, phone_names, len(grid_ids) - skipped, len(grid_ids)


def gather_layer(frames, lengths, rows, cols, layer):
    """[F, D] for one RVQ depth, reading only the valid frames of each clip."""
    out = np.empty((len(rows), frames.shape[2]), dtype=np.float32)
    # frames is [clips, layers, dim, max_frames]; index clip-major to avoid
    # materialising every layer.
    for r in np.unique(rows):
        mask = np.nonzero(rows == r)[0]
        c = cols[mask]
        out[mask] = np.take(frames[r, layer], c, axis=-1).T  # [dim, n] -> [n, dim]
    return out


def train_probe(x_train, y_train, x_val, y_val, num_classes, config, device="cpu"):
    model = nn.Linear(x_train.shape[1], num_classes).to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = train.build_optimizer(config, model)
    regularizer = train.build_regularizer(config)
    generator = torch.Generator().manual_seed(config.seed)

    for _ in range(config.epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), config.batch_size):
            idx = order[start : start + config.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_train[idx]), y_train[idx])
            if regularizer is not None:
                loss = loss + regularizer(model)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = torch.cat(
            [model(x_train[i : i + 4096]).argmax(1) for i in range(0, len(x_train), 4096)]
        )
        val_pred = torch.cat(
            [model(x_val[i : i + 4096]).argmax(1) for i in range(0, len(x_val), 4096)]
        )
    return {
        "val_acc": (val_pred == y_val).float().mean().item(),
        "train_acc": (pred == y_train).float().mean().item(),
        "pred": val_pred,
    }


def per_class_recall(pred, target, names, top=8):
    out = []
    for k in range(len(names)):
        mask = target == k
        if mask.sum() == 0:
            continue
        out.append((names[k], int(mask.sum()), (pred[mask] == k).float().mean().item()))
    out.sort(key=lambda r: -r[1])
    return out[:top]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--n-q", type=int, default=TrainConfig.n_q)
    parser.add_argument("--device", default=None,
                        help="torch device (cuda/cpu/mps); auto if omitted")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}")

    config = TrainConfig()
    config.epochs, config.batch_size, config.lr = args.epochs, args.batch_size, args.lr

    started = time.time()
    meta, rows, cols, frame_labels, speaker, phone_names, kept, total = load_frames(args.data_dir)
    print(f"clips usable: {kept}/{total} aligned | labelled frames: {len(frame_labels):,} "
          f"| phones: {len(phone_names)}")
    print(f"loaded in {time.time() - started:.1f}s")

    # Split whole speakers out, stratified only by which speakers exist.
    speakers = sorted(set(speaker.tolist()))
    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    cut = int(len(speakers) * (1 - args.val_fraction))
    val_speakers = set(speakers[cut:])
    is_val = np.array([s in val_speakers for s in speaker.tolist()])
    print(f"speakers: {len(speakers) - len(val_speakers)} train / {len(val_speakers)} held out")

    counts = np.bincount(frame_labels, minlength=len(phone_names))
    baseline = counts.max() / counts.sum()
    print(f"majority-class baseline: {baseline:.1%} ({phone_names[int(counts.argmax())]})")

    y = torch.from_numpy(frame_labels)
    results = {}
    for layer in range(args.n_q):
        x = gather_layer(meta["frames"], meta["lengths"], rows, cols, layer)
        mean = x[~is_val].mean(axis=0)
        std = x[~is_val].std(axis=0) + 1e-6
        x = torch.from_numpy((x - mean) / std)
        x_train, y_train = x[~is_val], y[~is_val]
        x_val, y_val = x[is_val], y[is_val]

        out = train_probe(x_train, y_train, x_val, y_val, len(phone_names), config, device)
        results[layer] = out
        print(f"  RVQ {layer + 1} | train {out['train_acc']:.3f} | "
              f"held-out speaker acc {out['val_acc']:.3f} "
              f"({out['val_acc'] / baseline:.1f}x baseline)")

    print("\nper-phone recall at the deepest layer (held-out speakers):")
    for name, n, recall in per_class_recall(results[args.n_q - 1]["pred"], y[is_val], phone_names):
        print(f"  {name:4s} {n:6d} frames  recall {recall:.2f}")

    print("\nsummary  (phone accuracy excluding silence, held-out speakers)")
    for layer, out in results.items():
        print(f"  RVQ {layer + 1}: {out['val_acc']:.3f}")


if __name__ == "__main__":
    main()