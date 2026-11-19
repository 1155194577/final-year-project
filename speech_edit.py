"""Speech editing: learn a male -> female direction and apply it to a man's voice.

Two spaces are involved and they are NOT the same:

  encoder space  what `model.encoder(wav)` returns (the residual input `e`)
  code space     codebook entries + `project_out`, summed over RVQ layers
                 (what `model.decoder(...)` consumes)

A direction learned on pooled *code-space* vectors only edits code space, so it
is applied there: the first RVQ layer's contribution is shifted for every frame,
and the decoder reconstructs audio from the edited sum.

The direction is the mean female minus mean male vector, computed on speaker-
held-out data with the mean removed, then applied at a chosen gain:

    id = decode(sum_k q_k)                    # the man, unchanged
    wf = decode(sum_k q_k + gain * v)         # the same man, moved toward female

    .venv/bin/python speech_edit.py
    .venv/bin/python speech_edit.py --gain 2.0 --text "..." --out figs/edit.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from backbone import load_model, load_audio

DATA_DIR = Path(__file__).parent / "data" / "libritts_preprocessed" / "dev.clean_36speakers_20utts"
FIG_PATH = Path(__file__).parent / "figs" / "speech_edit.png"


def code_space_frames(model, waveform, n_q=None):
    """[dim, T] cumulative RVQ sum and the per-layer quantized frames."""
    codes = model.encode(waveform[None, None])[:, 0, :]   # [layers, T]
    layers = model.quantizer.vq.layers
    per_layer = []
    total = None
    for k in range(n_q or codes.shape[0]):
        q = layers[k]
        vec = q.project_out(F.embedding(codes[k], q.codebook))   # [T, dim]
        per_layer.append(vec)
        total = vec if total is None else total + vec
    return total.T, per_layer


def apply_direction(total, per_layer, direction, layers=None, gain=1.0):
    """Add ``gain`` copies of ``direction``, confined to ``layers`` if given.

    The direction is rescaled per layer to that layer's own typical magnitude,
    so a fixed gain means the same relative perturbation at every depth.
    """
    shifted = total.clone()
    targets = range(len(per_layer)) if layers is None else [k for k in layers]
    unit = direction / direction.norm()
    for k in targets:
        magnitude = per_layer[k].norm(dim=1).median()
        shifted = shifted + gain * (unit * magnitude)[:, None]
    return shifted


def decode_sum(model, total, scale=1.0):
    """Decode a continuous [dim, T] quantized sum back to a waveform."""
    with torch.no_grad():
        out = model.decoder((total * scale)[None])     # [1, dim, T]
    return out[0, 0]


def fit_direction(model, meta, speakers, gender_of, n_q=None, layers=None):
    """Mean female minus mean male in code space, speaker-pooled, zero-mean.

    ``layers`` selects which RVQ layers form the representation the direction
    lives in: None means the full cumulative sum, a list means only those
    layers' contribution (e.g. [1..7] for "RVQ2-8", excluding the RVQ1 part
    that carries the phonetic content).
    """
    male, female = [], []
    for entry in meta["files"]:
        spk = str(entry["speaker_id"])
        if spk not in speakers:
            continue
        wav = load_audio(DATA_DIR / entry["file"], model.sample_rate)
        total, per = code_space_frames(model, wav, n_q)
        if layers is not None:
            # Sum the selected layers only: RVQ is residual, so layers 2..8 are
            # the residuals left after RVQ1 has coded the phonetic content.
            total = sum(per[k] for k in layers).T
        vec = total.mean(dim=1)                     # [dim], mean over time
        (female if gender_of[spk] == "female" else male).append(vec)
    mu_m = torch.stack(male).mean(0)
    mu_f = torch.stack(female).mean(0)
    direction = mu_f - mu_m
    # Normalise to the typical frame magnitude in this space: a unit vector is
    # negligible against per-frame vectors of norm ~150.
    typical = torch.stack([v.norm() for v in (mu_m, mu_f)]).mean()
    return direction / direction.norm() * typical, mu_m, mu_f


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", default=None,
                        help="male speaker id to edit (default: first male in metadata)")
    parser.add_argument("--utterance", default=None, help="clip id to edit")
    parser.add_argument("--gain", type=float, nargs="+", default=[1.0, 2.0, 4.0],
                        help="multiples of the unit direction to add")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="RVQ layers (0-based) to confine the edit to, e.g. 2 3 4 5 6 7 8 "
                             "for RVQ2-8. Default: the full sum.")
    parser.add_argument("--per-speaker", type=int, default=4,
                        help="clips per speaker when fitting the direction")
    parser.add_argument("--out", type=Path, default=FIG_PATH)
    args = parser.parse_args()

    meta = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    model = load_model(None, "cpu")
    gender_of = {str(e["speaker_id"]): e["gender"] for e in meta["files"]}

    # Fit on a few clips per speaker; ~36 speakers keeps this fast on CPU.
    speakers = sorted(
        {str(e["speaker_id"]) for e in meta["files"] if gender_of[str(e["speaker_id"])] in ("male", "female")}
    )[:36]
    subset = {"files": []}
    counts = {s: 0 for s in speakers}
    for e in meta["files"]:
        spk = str(e["speaker_id"])
        if spk in counts and counts[spk] < args.per_speaker:
            subset["files"].append(e)
            counts[spk] += 1

    layer_idx = None if args.layers is None else [n - 1 for n in args.layers]
    direction, mu_m, mu_f = fit_direction(
        model, subset, set(speakers), gender_of, layers=layer_idx
    )
    print(f"direction fitted on {len(subset['files'])} clips from {len(speakers)} speakers")
    print(f"|mu_male| {mu_m.norm():.3f} |mu_female| {mu_f.norm():.3f} | |mu_f-mu_m| {(mu_f-mu_m).norm():.3f}")

    # Pick a male clip to edit.
    target = args.utterance
    if target is None:
        males = [e for e in meta["files"] if e["gender"] == "male"
                 and (args.speaker is None or str(e["speaker_id"]) == args.speaker)]
        entry = males[0]
        target = entry["id"]
    else:
        entry = next(e for e in meta["files"] if e["id"] == target)
    wav = load_audio(DATA_DIR / entry["file"], model.sample_rate)
    print(f"editing {target} (speaker {entry['speaker_id']}, {entry['gender']}, "
          f"{len(wav) / model.sample_rate:.2f}s)")

    total, per_layer = code_space_frames(model, wav)
    anchor = decode_sum(model, total)                      # the man, unchanged
    n = min(len(wav), len(anchor))
    print("round-trip: original vs decoded(unchanged) SDR",
          f"{10 * torch.log10(wav[:n].pow(2).sum() / (wav[:n] - anchor[:n]).pow(2).sum()):.2f} dB")

    clips = [("original", wav)]
    directions = []
    for gain in args.gain:
        # ``total`` is [dim, T]; add the same direction to every frame.
        shifted = apply_direction(total, per_layer, direction, layer_idx, gain)
        audio = decode_sum(model, shifted)
        clips.append((f"gain {gain:g}", audio))
        frame_dir = (shifted - total).T.mean(0)
        directions.append((gain, frame_dir))

    out_dir = args.out.parent / "edits"
    out_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    for name, audio in clips:
        path = out_dir / f"{target}_{name.replace(' ', '')}.wav"
        sf.write(path, audio.detach().numpy(), model.sample_rate)
        print(f"  wrote {path}")
    print(f"listening check: {out_dir}")


if __name__ == "__main__":
    main()