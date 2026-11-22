"""Run the official kNN-VC (Baas et al., Interspeech 2023) as a comparison baseline.

Converts a male source clip toward a female reference speaker using the published
WavLM-Large encoder + prematched HiFiGAN vocoder, then reports F0 and WavLM
speaker similarity so it can be compared with the direction-vector edit.

    .venv/bin/python knn_vc_run.py
    .venv/bin/python knn_vc_run.py --topk 8 --refs 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).parent
KNNVC = ROOT / "third_party" / "knn-vc"
sys.path.insert(0, str(KNNVC))

DATA = ROOT / "data" / "libritts_preprocessed" / "dev.clean_36speakers_20utts"
OUT = ROOT / "figs" / "knnvc"
SR = 16000


def load_models(device="cpu"):
    from wavlm.WavLM import WavLM, WavLMConfig
    from hifigan.models import Generator as HiFiGAN
    from hifigan.utils import AttrDict
    from matcher import KNeighborsVC

    ckpt = torch.load(KNNVC / "checkpoints" / "WavLM-Large.pt", map_location=device,
                      weights_only=False)
    cfg = WavLMConfig(ckpt["cfg"])
    wavlm = WavLM(cfg)
    wavlm.load_state_dict(ckpt["model"])
    wavlm = wavlm.to(device).eval()

    h = AttrDict(json.loads((KNNVC / "hifigan" / "config_v1_wavlm.json").read_text()))
    hifigan = HiFiGAN(h).to(device)
    state = torch.load(KNNVC / "checkpoints" / "prematch_g_02500000.pt",
                       map_location=device, weights_only=False)
    hifigan.load_state_dict(state["generator"])
    hifigan = hifigan.eval()
    hifigan.remove_weight_norm()
    return KNeighborsVC(wavlm, hifigan, h, device)


def load16k(path) -> torch.Tensor:
    """Load a clip as a (1, T) tensor at 16 kHz without torchaudio.load (needs torchcodec)."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.dim() > 1:
        x = x.mean(dim=1)
    if sr != SR:
        x = torch.nn.functional.interpolate(
            x[None, None], size=int(round(len(x) * SR / sr)), mode="linear")[0, 0]
    return x[None]


def f0_estimate(x, sr=SR, fmin=60, fmax=400):
    """Median F0 via autocorrelation on voiced frames."""
    win, hop, out = 640, 160, []
    for i in range(0, len(x) - win, hop):
        frame = x[i:i + win] - x[i:i + win].mean()
        if np.sqrt((frame ** 2).mean()) < 0.01:
            continue
        ac = np.correlate(frame, frame, "full")[win - 1:]
        lo, hi = int(sr / fmax), int(sr / fmin)
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.3 * ac[0]:
            out.append(sr / lag)
    return float(np.median(out)) if out else float("nan")


def speaker_similarity(model, wav_a, wav_b):
    """Cosine between mean-pooled WavLM layer-6 features (a cheap speaker proxy)."""
    fa = model.get_features(torch.as_tensor(wav_a, dtype=torch.float32)[None]).mean(0)
    fb = model.get_features(torch.as_tensor(wav_b, dtype=torch.float32)[None]).mean(0)
    return torch.nn.functional.cosine_similarity(fa, fb, dim=0).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="174_50561_000023_000000")
    parser.add_argument("--ref-speaker", default=None, help="female speaker id for the reference")
    parser.add_argument("--refs", type=int, default=3, help="reference clips to use")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    meta = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))["files"]
    src = next(e for e in meta if e["id"] == args.source)

    ref_spk = args.ref_speaker
    if ref_spk is None:
        ref_spk = next(e["speaker_id"] for e in meta if e["gender"] == "female")
    refs = [e for e in meta if str(e["speaker_id"]) == str(ref_spk)][: args.refs]
    print(f"source : {src['id']} (speaker {src['speaker_id']}, {src['gender']})")
    print(f"reference: speaker {ref_spk} ({refs[0]['gender']}), {len(refs)} clips")

    model = load_models(args.device)
    src_wav, sr = sf.read(DATA / src["file"], dtype="float32")
    if sr != SR:
        import torchaudio
        src_wav = torchaudio.functional.resample(
            torch.as_tensor(src_wav)[None], sr, SR)[0].numpy()

    query = model.get_features(load16k(DATA / src["file"]), vad_trigger_level=7)
    bank = model.get_matching_set([load16k(DATA / r["file"]) for r in refs], vad_trigger_level=7)
    print(f"query frames {tuple(query.shape)} | reference bank {tuple(bank.shape)}")

    out = model.match(query, bank, topk=args.topk, tgt_loudness_db=-16).cpu().numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(OUT / f"{src['id']}_knnvc.wav", out, SR)
    print(f"wrote {OUT / (src['id'] + '_knnvc.wav')} ({len(out)/SR:.2f}s)")

    ref_wav, _ = sf.read(DATA / refs[0]["file"], dtype="float32")
    if len(ref_wav) > SR * 4:
        ref_wav = ref_wav[: SR * 4]

    print("\n                    F0 (Hz)   WavLM sim to reference")
    print(f"  original        {f0_estimate(src_wav):7.1f}   "
          f"{speaker_similarity(model, src_wav, ref_wav):.3f}")
    print(f"  kNN-VC output   {f0_estimate(out):7.1f}   "
          f"{speaker_similarity(model, out, ref_wav):.3f}")

    edit = ROOT / "figs" / "edits" / f"{src['id']}_gain1.wav"
    if edit.exists():
        e, esr = sf.read(edit, dtype="float32")
        print(f"  direction edit  {f0_estimate(e):7.1f}   "
              f"{speaker_similarity(model, e, ref_wav):.3f}   ({esr} Hz)")


if __name__ == "__main__":
    main()