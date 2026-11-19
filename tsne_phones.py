"""t-SNE of SpeechTokenizer frames coloured by MFA phone label.

Two panels, same frames:

  left   RVQ-1  (the shallowest representation)
  right  RVQ-8  (full cumulative prefix)

Phones are grouped into broad manners of articulation so the picture is legible;
the probe itself predicts all 39 phones. Frames are subsampled to keep t-SNE
fast, and the sample is stratified by phone so rare phones still appear.

    .venv/bin/python tsne_phones.py
    .venv/bin/python tsne_phones.py --per-phone 60 --layer 0
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

import phone_probe as pp

DATA_DIR = Path(__file__).parent / "data" / "libritts_preprocessed" / "dev.clean_36speakers_20utts"
FIG_PATH = Path(__file__).parent / "figs" / "tsne_phones.png"

# ARPAbet phones grouped by manner of articulation.
GROUPS = {
    "vowel": ["aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"],
    "stop": ["b", "d", "g", "p", "t", "k"],
    "fricative": ["f", "v", "th", "dh", "s", "z", "sh", "zh", "hh"],
    "affricate": ["ch", "jh"],
    "nasal": ["m", "n", "ng"],
    "liquid": ["l", "r"],
    "glide": ["w", "y"],
}
COLORS = {
    "vowel": "#d1495b", "stop": "#2e86ab", "fricative": "#edae49",
    "affricate": "#9b5de5", "nasal": "#00a878", "liquid": "#f15bb5", "glide": "#8d99ae",
}


def group_of(phone):
    for name, members in GROUPS.items():
        if phone in members:
            return name
    return None


def stratified_sample(y, per_phone, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for phone in np.unique(y):
        idx = np.flatnonzero(y == phone)
        take = min(per_phone, idx.size)
        keep.append(rng.choice(idx, take, replace=False))
    return np.sort(np.concatenate(keep))


def embed(x, seed=0):
    # Subsample further for t-SNE itself if needed; it is O(n log n) but slow above ~15k.
    return TSNE(
        n_components=2, perplexity=30, init="pca", random_state=seed,
        max_iter=750, verbose=0,
    ).fit_transform(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--per-phone", type=int, default=150,
                        help="frames sampled per phone before embedding")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIG_PATH)
    parser.add_argument("--layers", type=int, nargs=2, default=(0, 7),
                        help="two RVQ depths to compare")
    args = parser.parse_args()

    meta, rows, cols, y, spk, names, kept, total = pp.load_frames(args.data_dir)
    sample = stratified_sample(y, args.per_phone, args.seed)
    print(f"sampled {len(sample):,} frames from {len(y):,} ({args.per_phone} per phone)")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    for ax, layer in zip(axes, args.layers):
        x = pp.gather_layer(meta["frames"], meta["lengths"], rows[sample], cols[sample], layer)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        if len(x) > 15000:                       # keep t-SNE tractable
            sub = np.random.default_rng(args.seed).choice(len(x), 15000, replace=False)
            x, y_sub = x[sub], y[sample][sub]
        else:
            y_sub = y[sample]
        z = embed(x, args.seed)

        groups = np.array([group_of(names[p]) or "other" for p in y_sub])
        for group, color in COLORS.items():
            mask = groups == group
            if not mask.any():
                continue
            ax.scatter(z[mask, 0], z[mask, 1], s=7, alpha=0.6, color=color,
                       label=f"{group} ({mask.sum()})", linewidths=0)
        ax.set_title(f"RVQ-{layer + 1}", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)

    axes[0].legend(loc="upper right", fontsize=8, frameon=False, markerscale=2.2)
    fig.suptitle("SpeechTokenizer frames coloured by MFA phone (grouped by manner of articulation)",
                 fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()