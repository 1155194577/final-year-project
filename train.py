"""Minimal training loop: train one linear probe per feature layer.

Features come from ``SpeechTokenizerFeatureExtractor`` (frozen SpeechTokenizer),
which caches a ``[num_utterances, num_layers, dimension]`` bundle under
``<dataset>/features/features.npz``. Layer ``n`` of that bundle is the input to
probe ``n``.

Run with the project interpreter:
    .venv/bin/python train.py
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import TrainConfig
from data_preprocessor import PreparedDataset
from model import LinearLayer
from speechtokenizer_features import SpeechTokenizerFeatureExtractor


def get_bundle(config):
    """Load the cached feature bundle, or extract (and cache) it on first run."""
    data_dir = Path(config.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent / data_dir
    dataset = PreparedDataset.load(data_dir)
    extractor = SpeechTokenizerFeatureExtractor(
        model_dir=config.model_dir,
        device=config.device,
        layers=list(range(config.n_q)),
        normalize=True,
    )
    output_dir = Path(config.features_dir) if config.features_dir else None
    bundle = extractor.extract(dataset, output_dir=output_dir)
    print(bundle.summary())
    return bundle


def get_data(config, bundle, layer):
    """Return train/val DataLoaders of (features, speaker label) for one layer."""
    speakers = sorted(set(bundle.speaker_ids.tolist()))
    label_of = {str(spk): i for i, spk in enumerate(speakers)}
    if len(speakers) > config.num_of_class:
        raise ValueError(
            f"{len(speakers)} speakers but num_of_class={config.num_of_class}"
        )

    features = torch.from_numpy(bundle.features[:, layer, :]).float()
    labels = torch.tensor(
        [label_of[str(spk)] for spk in bundle.speaker_ids.tolist()],
        dtype=torch.long,
    )

    rng = random.Random(config.seed)
    train_idx, val_idx = [], []
    if config.split == "speaker":
        # Hold out whole speakers: the probe never sees the val voices, so the
        # accuracy measures generalisation rather than speaker memorisation.
        order = list(range(len(speakers)))
        rng.shuffle(order)
        cut = int(len(order) * (1 - config.val_fraction))
        held_out = set(order[cut:])
        for i, label in enumerate(labels.tolist()):
            (val_idx if label in held_out else train_idx).append(i)
    else:
        # Split each speaker's utterances: every speaker appears in both sets.
        for spk in range(len(speakers)):
            idx = torch.nonzero(labels == spk).flatten().tolist()
            rng.shuffle(idx)
            cut = int(len(idx) * (1 - config.val_fraction))
            train_idx += idx[:cut]
            val_idx += idx[cut:]

    # Standardize on training statistics.
    mean, std = features[train_idx].mean(dim=0), features[train_idx].std(dim=0) + 1e-6
    features = (features - mean) / std

    train = TensorDataset(features[train_idx], labels[train_idx])
    val = TensorDataset(features[val_idx], labels[val_idx])
    return (
        DataLoader(train, batch_size=config.batch_size, shuffle=True),
        DataLoader(val, batch_size=config.batch_size),
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += len(y)
    return total_loss / count, correct / count


def build_criterion(config):
    if config.criterion == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"unknown criterion: {config.criterion}")


def build_regularizer(config):
    """Return a callable computing the penalty for the model, or None."""
    name = (config.regularizer or "none").lower()
    if name in ("none", "off", ""):
        return None
    if name not in ("l1", "l2"):
        raise ValueError(f"unknown regularizer: {config.regularizer}")

    def penalty(model):
        params = [p for p in model.parameters() if p.requires_grad]
        if name == "l1":
            total = sum(p.abs().sum() for p in params)
        else:
            total = sum(p.pow(2).sum() for p in params)
        return config.reg_weight * total

    return penalty


def build_optimizer(config, model):
    name = config.optimizer.lower()
    if name in ("sdg", "sgd"):  # "sdg" is a typo that lives in some configs
        return torch.optim.SGD(model.parameters(), lr=config.lr)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr)
    raise ValueError(f"unknown optimizer: {config.optimizer}")


def train_layer(config, train_loader, val_loader, layer, device):
    """Train a linear probe and return the best validation accuracy."""
    model = LinearLayer(config.embedding_vector_dim, config.num_of_class).to(device)
    criterion = build_criterion(config)
    optimizer = build_optimizer(config, model)
    regularizer = build_regularizer(config)

    best_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if regularizer is not None:
                loss = loss + regularizer(model)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        best_acc = max(best_acc, val_acc)
        print(
            f"  layer {layer} | epoch {epoch:2d}/{config.epochs} | "
            f"train loss {train_loss:.3f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.3f} acc {val_acc:.3f}"
        )
    return best_acc


def parse_args(argv=None):
    """Build a config from TrainConfig defaults, overridable from the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=TrainConfig.data_dir)
    parser.add_argument("--model-dir", default=TrainConfig.model_dir)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--features-dir", default=TrainConfig.features_dir)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--optimizer", default=TrainConfig.optimizer)
    parser.add_argument("--regularizer", default=TrainConfig.regularizer)
    parser.add_argument("--reg-weight", type=float, default=TrainConfig.reg_weight)
    parser.add_argument("--n-q", type=int, default=TrainConfig.n_q)
    parser.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    parser.add_argument("--split", choices=("utterance", "speaker"),
                        default=TrainConfig.split)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    config = TrainConfig()
    for key, value in vars(parser.parse_args(argv)).items():
        setattr(config, key, value)
    return config


def main(config):
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")
    print(f"regularizer: {config.regularizer} (weight {config.reg_weight})")

    bundle = get_bundle(config)
    if bundle.dimension != config.embedding_vector_dim:
        raise ValueError(
            f"bundle dimension {bundle.dimension} != "
            f"embedding_vector_dim {config.embedding_vector_dim}"
        )

    results = {}
    for n in range(config.n_q):
        print(f"probing now : the {n}th layer")
        train_loader, val_loader = get_data(config, bundle, n)
        results[n] = train_layer(config, train_loader, val_loader, n, device)

    print("\nbest val accuracy per layer:")
    for n, acc in results.items():
        print(f"  layer {n}: {acc:.3f}")


if __name__ == "__main__":
    main(parse_args())
