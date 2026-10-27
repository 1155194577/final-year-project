"""Gender probe on SpeechTokenizer RVQ embeddings, plus a t-SNE picture.

Two conditions, same 1024-D features, same protocol:

  rvq1     the first RVQ layer alone        (q_0)
  rvq2_8   the full 8-layer prefix          (q_0 + q_1 + ... + q_7)

The probe is trained on utterances from a subset of speakers and evaluated on
*held-out speakers* (stratified by gender), so the accuracy measures
generalisation to unseen voices rather than memorising the training speakers.

Run:
    .venv/bin/python gender_class.py
"""

from dataclasses import dataclass, field
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

import train
from config import TrainConfig
from model import LinearLayer

FIGS_DIR = Path(__file__).parent / "figs"
GENDERS = ("female", "male")


@dataclass
class GenderData:
    """Standardised features for one condition, with a speaker-level split."""

    name: str
    x: torch.Tensor  # [N, dim]
    y: torch.Tensor  # [N] 0=female, 1=male
    speaker_ids: np.ndarray
    train_idx: list[int]
    val_idx: list[int]
    predictions: torch.Tensor | None = field(default=None)

    @property
    def train_x(self):
        return self.x[self.train_idx]

    @property
    def val_x(self):
        return self.x[self.val_idx]

    @property
    def train_y(self):
        return self.y[self.train_idx]

    @property
    def val_y(self):
        return self.y[self.val_idx]


def _feature_path(config) -> Path:
    data_dir = Path(config.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent / data_dir
    return data_dir / "features" / "features.npz"


def _split_speakers(speaker_ids, genders, seed, val_fraction):
    """Hold out whole speakers, keeping the gender mix in both halves."""
    gender_of = {}
    for speaker, gender in zip(speaker_ids.tolist(), genders.tolist()):
        gender_of[speaker] = gender

    rng = random.Random(seed)
    train_speakers, val_speakers = set(), set()
    for gender in GENDERS:
        speakers = sorted(s for s, g in gender_of.items() if g == gender)
        rng.shuffle(speakers)
        cut = int(len(speakers) * (1 - val_fraction))
        train_speakers.update(speakers[:cut])
        val_speakers.update(speakers[cut:])

    train_idx = [i for i, s in enumerate(speaker_ids.tolist()) if s in train_speakers]
    val_idx = [i for i, s in enumerate(speaker_ids.tolist()) if s in val_speakers]
    return train_idx, val_idx


def _load_condition(layer, name, config=None):
    config = config or TrainConfig()
    bundle = np.load(_feature_path(config), allow_pickle=False)

    x = torch.from_numpy(bundle["features"][:, layer, :]).float()
    genders = bundle["genders"]
    speaker_ids = bundle["speaker_ids"]
    y = torch.tensor([0 if g == "female" else 1 for g in genders], dtype=torch.long)

    train_idx, val_idx = _split_speakers(speaker_ids, genders, config.seed, config.val_fraction)

    # Standardise on training speakers only.
    mean, std = x[train_idx].mean(dim=0), x[train_idx].std(dim=0) + 1e-6
    x = (x - mean) / std

    return GenderData(
        name=name, x=x, y=y, speaker_ids=speaker_ids,
        train_idx=train_idx, val_idx=val_idx,
    )


def get_rvq1_vector(config=None):
    """RVQ layer 1 only: the first quantizer's cumulative codebook vector."""
    return _load_condition(layer=0, name="rvq1", config=config)


def get_rvq2_8_vector(config=None):
    """RVQ layers 2..8: the 8-layer prefix, i.e. the deepest representation."""
    return _load_condition(layer=7, name="rvq2_8", config=config)


def train_classification(data, config=None):
    """Train a linear probe on `data.train_*`, return held-out accuracy."""
    config = config or TrainConfig()
    model = LinearLayer(data.x.shape[1], len(GENDERS))
    criterion = train.build_criterion(config)
    optimizer = train.build_optimizer(config, model)
    regularizer = train.build_regularizer(config)

    x_train, y_train = data.train_x, data.train_y
    generator = torch.Generator().manual_seed(config.seed)

    for _ in range(config.epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_train[batch]), y_train[batch])
            if regularizer is not None:
                loss = loss + regularizer(model)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        data.predictions = model(data.x).argmax(dim=1)

    correct = (data.predictions[data.val_idx] == data.val_y).float().mean().item()
    for label, gender in enumerate(GENDERS):
        mask = data.val_y == label
        per_gender = (data.predictions[data.val_idx][mask] == label).float().mean().item()
        print(f"  {gender:6s}: {per_gender:.3f} ({int(mask.sum())} held-out utterances)")
    return correct


def plot_tsne(features, prediction, ground_truth, held_out, title, out_path):
    """Scatter a t-SNE of the embeddings, coloured by gender."""
    embedding = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=0,
        max_iter=1000,
    ).fit_transform(features.numpy())

    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = ("#d1495b", "#2e86ab")  # female, male
    for label, gender in enumerate(GENDERS):
        mask = (ground_truth == label).numpy()
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=26, color=colors[label], alpha=0.75,
            edgecolor="none", label=f"{gender} (n={int(mask.sum())})",
        )

    wrong = (prediction != ground_truth).numpy() & held_out.numpy()
    ax.scatter(
        embedding[wrong, 0], embedding[wrong, 1],
        s=54, facecolor="none", edgecolor="black", linewidths=1.2,
        label=f"misclassified held-out utterance (n={int(wrong.sum())})",
    )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main():
    config = TrainConfig()
    print(f"epochs={config.epochs} batch={config.batch_size} lr={config.lr} "
          f"optimizer={config.optimizer} regularizer={config.regularizer}")

    for getter in (get_rvq1_vector, get_rvq2_8_vector):
        data = getter(config)
        train_speakers = {data.speaker_ids[i] for i in data.train_idx}
        val_speakers = {data.speaker_ids[i] for i in data.val_idx}
        print(f"\n{data.name}: {len(data.train_idx)} train / {len(data.val_idx)} val "
              f"utterances ({len(train_speakers)} / {len(val_speakers)} speakers)")

        accuracy = train_classification(data, config)
        print(f"  held-out speaker accuracy: {accuracy:.3f}")

        held_out = torch.zeros(len(data.y), dtype=torch.bool)
        held_out[data.val_idx] = True
        plot_tsne(
            data.x, data.predictions, data.y, held_out,
            title=f"t-SNE of {data.name} embeddings (colour = gender, "
                  f"held-out speaker acc = {accuracy:.3f})",
            out_path=FIGS_DIR / f"tsne_gender_{data.name}.png",
        )


if __name__ == "__main__":
    main()
