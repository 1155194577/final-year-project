"""Fit a lightweight affine map (x -> a*x + b) from one male speaker to one female speaker.

Correspondence is established through the MFA phone labels: for every phone class
present in both speakers we take the mean frame vector, giving (male, female)
pairs. A scalar gain ``a`` and a bias vector ``b`` are then fitted in closed form
by least squares, and every male frame is transformed before decoding.

    .venv/bin/python affine_vc.py --male 174 --female 84
    .venv/bin/python affine_vc.py --mode diag        # per-dimension gain
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from backbone import load_model, load_audio
from speech_edit import code_space_frames, decode_sum

DATA = Path(__file__).parent / "data" / "libritts_preprocessed" / "dev.clean_36speakers_20utts"
OUT = Path(__file__).parent / "figs" / "affine"
SR = 16000


def phone_means(per_frame, labels, n_phones, layer):
    """Mean frame vector per phone class for one speaker."""
    means = {}
    for p in range(n_phones):
        idx = np.flatnonzero(labels == p)
        if len(idx) < 5:                       # too few frames to be a stable mean
            continue
        means[p] = per_frame[idx].mean(axis=0)
    return means


def fit_scalar(pairs_m, pairs_f):
    """Least squares for x' = a*x + b with scalar a and free b."""
    M, F = torch.stack(pairs_m), torch.stack(pairs_f)
    mc, fc = M - M.mean(0), F - F.mean(0)
    a = (mc * fc).sum() / (mc * mc).sum()
    b = F.mean(0) - a * M.mean(0)
    resid = (a * M + b - F).norm(dim=1).mean() / F.norm(dim=1).mean()
    return a.item(), b, resid.item()


def fit_diag(pairs_m, pairs_f):
    """Per-dimension gain (1024 params) with a shared bias: x' = a_d*x_d + b."""
    M, F = torch.stack(pairs_m), torch.stack(pairs_f)
    mc, fc = M - M.mean(0), F - F.mean(0)
    a = (mc * fc).sum(0) / (mc * mc).sum(0).clamp(min=1e-8)
    b = F.mean(0) - a * M.mean(0)
    resid = (a * M + b - F).norm(dim=1).mean() / F.norm(dim=1).mean()
    return a, b, resid.item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--male", default="174")
    ap.add_argument("--female", default="84")
    ap.add_argument("--layer", type=int, default=7, help="RVQ depth to edit (1-based), default 8")
    ap.add_argument("--mode", choices=("scalar", "diag"), default="scalar")
    ap.add_argument("--clips", type=int, default=20, help="clips per speaker")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    meta = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))["files"]
    phones = np.load(DATA / "features" / "phones.npz", allow_pickle=False)
    label_of = {str(u): i for i, u in enumerate(phones["utterance_ids"])}
    n_phones = len(phones["phone_names"])

    male = [e for e in meta if str(e["speaker_id"]) == args.male][: args.clips]
    female = [e for e in meta if str(e["speaker_id"]) == args.female][: args.clips]
    if not male or not female:
        raise SystemExit("speaker ids not found in metadata")
    print(f"male speaker {args.male}: {len(male)} clips | "
          f"female speaker {args.female}: {len(female)} clips")

    model = load_model(None, args.device)
    layer = args.layer - 1

    def collect(entries):
        frames, labels = [], []
        for e in entries:
            if e["id"] not in label_of:
                continue
            wav = load_audio(DATA / e["file"], SR)
            total, per = code_space_frames(model, wav)
            # cumulative prefix up to `layer`
            cum = sum(per[: layer + 1])                        # [T, dim]
            n = min(len(cum), int(phones["lengths"][label_of[e["id"]]]))
            frames.append(cum[:n].cpu().numpy())
            labels.append(phones["labels"][label_of[e["id"]]][:n])
        return np.concatenate(frames), np.concatenate(labels)

    print("encoding male clips ...")
    Mx, My = collect(male)
    print("encoding female clips ...")
    Fx, Fy = collect(female)
    print(f"frames: male {Mx.shape[0]:,} | female {Fx.shape[0]:,}")

    mM = phone_means(Mx, My, n_phones, layer)
    mF = phone_means(Fx, Fy, n_phones, layer)
    shared = sorted(set(mM) & set(mF))
    print(f"shared phone classes for correspondence: {len(shared)}")

    pairs_m = [torch.from_numpy(mM[p]) for p in shared]
    pairs_f = [torch.from_numpy(mF[p]) for p in shared]

    if args.mode == "scalar":
        a, b, resid = fit_scalar(pairs_m, pairs_f)
        print(f"fitted a = {a:.4f}, |b| = {b.norm():.2f}, relative residual {resid:.1%}")
        transform = lambda x: a * x + b
    else:
        a, b, resid = fit_diag(pairs_m, pairs_f)
        print(f"fitted per-dim gain: mean {a.mean():.4f}, |b| = {b.norm():.2f}, "
              f"relative residual {resid:.1%}")
        transform = lambda x: a * x + b

    # How much does the fitted map actually move a frame, relative to its norm?
    Mx_t = torch.from_numpy(Mx)
    moved = transform(Mx_t) - Mx_t
    disp = (moved.norm(dim=1) / Mx_t.norm(dim=1)).median().item()
    # ...and how much of that movement is in the male->female direction?
    fem = phone_means(Fx, Fy, n_phones, layer)
    mal = phone_means(Mx, My, n_phones, layer)
    v = torch.stack([torch.from_numpy(fem[p]) - torch.from_numpy(mal[p]) for p in shared]).mean(0)
    v = v / v.norm()
    along = (moved @ v).median().item()
    print(f"median relative displacement |x'-x|/|x| = {disp:.3f} | "
          f"component along male->female axis = {along:+.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    src = male[0]
    wav = load_audio(DATA / src["file"], SR)
    total, per = code_space_frames(model, wav)
    cum = sum(per[: layer + 1])                                # [T, dim]
    with torch.no_grad():
        edited = transform(cum)                                # [T, dim]
        out = model.decoder(edited.T[None])[0, 0].cpu().numpy()   # decoder wants [1, dim, T]

    label = f"a{a:.3f}" if args.mode == "scalar" else "diag"
    path = OUT / f"{src['id']}_{args.mode}_{label}.wav"
    sf.write(path, out, SR)
    print(f"wrote {path} ({len(out)/SR:.2f}s)")


if __name__ == "__main__":
    main()