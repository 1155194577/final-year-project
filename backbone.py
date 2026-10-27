"""Minimal frozen SpeechTokenizer backbone: device, model download, audio loading.

Everything here is plain library code, so feature extraction does not depend on
any external experiment module. The checkpoint is the official SpeechTokenizer
release (the old ``fnlp/SpeechTokenizer`` repo now redirects here).
"""

from __future__ import annotations

import urllib.request
import wave
from pathlib import Path

import numpy as np
import torch

MODEL_REPO = "OpenMOSS-Team/SpeechTokenizer"
MODEL_SUBDIR = "speechtokenizer_hubert_avg"
MODEL_FILES = ("config.json", "SpeechTokenizer.pt")
DEFAULT_MODEL_DIR = Path(__file__).parent / "data" / "SpeechTokenizer"
USER_AGENT = "fyp-speechtokenizer-features/1.0"


def choose_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    print(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as output:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                output.write(chunk)
    temporary.replace(destination)
    print(f"wrote {destination} ({destination.stat().st_size / 1e6:.0f} MB)")


def resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    """Return the directory holding ``config.json`` + ``SpeechTokenizer.pt``.

    Downloads the checkpoint on first use when it is not already present.
    """
    root = Path(model_dir).expanduser() if model_dir else DEFAULT_MODEL_DIR
    if not root.is_absolute():
        root = Path(__file__).parent / root
    # Accept either the snapshot root or the checkpoint subdirectory itself.
    candidate = root if (root / MODEL_FILES[1]).exists() else root / MODEL_SUBDIR
    if all((candidate / name).exists() for name in MODEL_FILES):
        return candidate

    for name in MODEL_FILES:
        destination = candidate / name
        if not destination.exists():
            _download(
                f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_SUBDIR}/{name}",
                destination,
            )
    return candidate


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Load a mono waveform at ``sample_rate`` (1-D float tensor)."""
    path = Path(path)
    try:
        import soundfile
    except ImportError:
        soundfile = None

    if soundfile is not None:
        audio, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio).mean(dim=1)
    else:
        with wave.open(str(path), "rb") as handle:
            if handle.getsampwidth() != 2:
                raise RuntimeError(
                    f"{path}: only 16-bit PCM is supported without soundfile"
                )
            channels, sr = handle.getnchannels(), handle.getframerate()
            frames = handle.readframes(handle.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        waveform = torch.from_numpy(audio)

    if sr != sample_rate:
        new_length = int(round(len(waveform) * sample_rate / sr))
        waveform = torch.nn.functional.interpolate(
            waveform[None, None], size=new_length, mode="linear"
        )[0, 0]
    return waveform


def load_model(model_dir: str | Path, device: torch.device | None = None):
    """Load a frozen SpeechTokenizer model."""
    from speechtokenizer import SpeechTokenizer

    directory = resolve_model_dir(model_dir)
    model = SpeechTokenizer.load_from_checkpoint(
        str(directory / "config.json"), str(directory / "SpeechTokenizer.pt")
    )
    model = model.to(device or choose_device()).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
