"""Compare the mean-difference gender direction with a shrunk-LDA direction.

Both directions are fitted on the same pooled SpeechTokenizer features, then
applied in code space to the same male clip at matched displacement, decoded,
and scored with F0 plus WavLM frame-wise speaker similarity.

The mean-difference direction ignores within-class covariance:

    v_mean = mu_female - mu_male

whereas Fisher LDA whitens by it first:

    v_lda = Sigma_W^-1 (mu_female - mu_male)

with shrinkage so the 1024-d covariance is invertible from 720 clips.

    .venv/bin/python lda_direction.py
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from backbone import load_model, load_audio
from speech_edit import code_space_frames, decode_sum

DATA = Path(__file__).parent / "data" / "libritts_preprocessed" / "dev.clean_36speakers_20utts"
OUT = Path(__file__).parent / "figs" / "lda"
SR = 16000


def shrunk_within_covariance(X, groups, alpha):
    """Pooled within-group covariance with shrinkage toward sigma^2 I."""
    d = X.shape[1]
    cov, n = np.zeros((d, d)), 0
    for g in np.unique(groups):
        Z = X[groups == g] - X[groups == g].mean(0)
        cov += Z.T @ Z
        n += len(Z)
    cov /= n
    sigma2 = np.trace(cov) / d
    return (1 - alpha) * cov + alpha * sigma2 * np.eye(d)


def main():
    bundle = np.load(DATA / "features" / "features.npz", allow_pickle=False)
    X = bundle["features"][:, 7, :].astype(np.float64)      # cumulative prefix, L2-normalised
    y = bundle["genders"]
    print(f"features {X.shape} | female {(y=='female').sum()} male {(y=='male').sum()}")

    mu_m, mu_f = X[y == "male"].mean(0), X[y == "female"].mean(0)
    v_mean = mu_f - mu_m

    for alpha in (0.3, 0.1):
        sw = shrunk_within_covariance(X, y, alpha)
        v_lda = np.linalg.solve(sw, v_mean)
        cos = float(v_lda @ v_mean / (np.linalg.norm(v_lda) * np.linalg.norm(v_mean)))
        acc = float(((X @ v_lda > np.median(X @ v_lda)) == (y == "female")).mean())
        acc_m = float(((X @ v_mean > np.median(X @ v_mean)) == (y == "female")).mean())
        print(f"\nshrinkage {alpha}: angle(v_mean, v_lda) = {np.degrees(np.arccos(cos)):.1f} deg")
        print(f"  gender accuracy  v_mean {max(acc_m, 1-acc_m):.1%} | v_lda {max(acc, 1-acc):.1%}")

        if alpha != 0.1:
            continue

        model = load_model(None, "cpu")
        meta = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))["files"]
        src = next(e for e in meta if e["id"] == "174_50561_000023_000000")
        wav = load_audio(DATA / src["file"], SR)
        total, _ = code_space_frames(model, wav)
        magnitude = total.norm(dim=0).median()

        OUT.mkdir(parents=True, exist_ok=True)
        for name, vec in (("mean", v_mean), ("lda", v_lda)):
            unit = torch.from_numpy(vec / np.linalg.norm(vec)).float()
            shifted = total + magnitude * unit[:, None]
            with torch.no_grad():
                audio = model.decoder(shifted[None])[0, 0].numpy()
            path = OUT / f"{src['id']}_{name}.wav"
            sf.write(path, audio, SR)
            print(f"  wrote {path}")

        # diagnostics on the decoded audio
        import knn_vc_run as K
        refs = [e for e in meta if str(e["speaker_id"]) == "84"][:3]
        wavlm = K.load_models("cpu")

        def feats(path):
            x, sr = sf.read(str(path), dtype="float32")
            if sr != 16000:
                x = torch.nn.functional.interpolate(
                    torch.as_tensor(x)[None, None], size=int(len(x) * 16000 / sr),
                    mode="linear")[0, 0].numpy()
            return wavlm.get_features(torch.as_tensor(x, dtype=torch.float32)[None])

        def framewise(a, b, n=200):
            a = torch.nn.functional.normalize(a, dim=1)
            b = torch.nn.functional.normalize(b, dim=1)
            A = a[torch.linspace(0, len(a) - 1, min(n, len(a))).long()]
            B = b[torch.linspace(0, len(b) - 1, min(n, len(b))).long()]
            return 0.5 * ((A @ B.T).max(1).values.mean() + (B @ A.T).max(1).values.mean()).item()

        def centroid(path):
            x, sr = sf.read(str(path), dtype="float32")
            Z = np.abs(np.fft.rfft(x[: 16000 * 4] * np.hanning(min(len(x), 16000 * 4))))
            f = np.fft.rfftfreq(min(len(x), 16000 * 4), 1 / sr)
            return float((Z * f).sum() / Z.sum())

        bank = torch.cat([feats(DATA / e["file"]) for e in refs])
        print(f"\n  {'variant':10s} {'F0':>6} {'sim->84':>8} {'centroid':>9}")
        rows = [("original", DATA / src["file"]), ("v_mean", OUT / f"{src['id']}_mean.wav"),
                ("v_lda", OUT / f"{src['id']}_lda.wav")]
        for name, p in rows:
            x, sr = sf.read(str(p), dtype="float32")
            if sr != 16000:
                x = torch.nn.functional.interpolate(
                    torch.as_tensor(x)[None, None], size=int(len(x) * 16000 / sr),
                    mode="linear")[0, 0].numpy()
            print(f"  {name:10s} {K.f0_estimate(x):6.1f} "
                  f"{framewise(feats(p), bank):8.3f} {centroid(p):9.0f}")


if __name__ == "__main__":
    main()