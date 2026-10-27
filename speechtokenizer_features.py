#!/usr/bin/env python3
"""Extract per-layer SpeechTokenizer RVQ embeddings for a prepared dataset.

The data download/selection is handled by ``DataPreprocessor``. This module
contains only frozen SpeechTokenizer inference and feature caching.

For every utterance and every RVQ layer:

1. encode audio into codes;
2. look up each frame's codebook vector;
3. apply the quantizer's output projection;
4. sum layers 0..k (RVQ layers are residual, so this prefix sum is what the
   decoder actually reconstructs at depth k);
5. mean-pool over time;
6. optionally L2-normalize.

Example:
    dataset = DataPreprocessor(...).prepare()
    extractor = SpeechTokenizerFeatureExtractor(model_dir="data/SpeechTokenizer")
    bundle = extractor.extract(dataset)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_preprocessor import PreparedDataset
from backbone import choose_device, load_audio, load_model, resolve_model_dir


@dataclass
class FeatureBundle:
    features: np.ndarray  # [num_utterances, num_layers, dimension]
    speaker_ids: np.ndarray
    genders: np.ndarray
    utterance_ids: np.ndarray
    slots: np.ndarray
    layer_indices: np.ndarray
    split: str
    root: Path

    @property
    def num_utterances(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_layers(self) -> int:
        return int(self.features.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.features.shape[2])

    def summary(self) -> dict[str, object]:
        return {
            "split": self.split,
            "num_utterances": self.num_utterances,
            "num_speakers": len(set(self.speaker_ids.tolist())),
            "num_layers": self.num_layers,
            "dimension": self.dimension,
            "layer_indices": self.layer_indices.tolist(),
            "root": str(self.root),
        }

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not np.all(np.isfinite(self.features)):
            problems.append("Non-finite features")
        if len(self.utterance_ids) != self.num_utterances:
            problems.append("Utterance ID count mismatch")
        if len(self.speaker_ids) != self.num_utterances:
            problems.append("Speaker ID count mismatch")
        if len(self.genders) != self.num_utterances:
            problems.append("Gender count mismatch")
        if len(self.slots) != self.num_utterances:
            problems.append("Slot count mismatch")
        return problems

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            speaker_ids=self.speaker_ids,
            genders=self.genders,
            utterance_ids=self.utterance_ids,
            slots=self.slots,
            layer_indices=self.layer_indices,
            split=np.asarray(self.split),
        )

    @classmethod
    def load(cls, path: Path, root: Path) -> "FeatureBundle":
        data = np.load(path, allow_pickle=False)
        return cls(
            features=data["features"],
            speaker_ids=data["speaker_ids"],
            genders=data["genders"],
            utterance_ids=data["utterance_ids"],
            slots=data["slots"],
            layer_indices=data["layer_indices"],
            split=str(data["split"]),
            root=root,
        )


class SpeechTokenizerFeatureExtractor:
    """Frozen SpeechTokenizer feature extraction."""

    def __init__(
        self,
        model_dir: Path | None = None,
        device: str | None = None,
        layers: list[int] | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_dir = model_dir
        self.requested_device = device
        self.layers = layers
        self.normalize = normalize
        self._model = None
        self._device = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        device = choose_device(self.requested_device)
        model_dir = resolve_model_dir(self.model_dir)
        model = load_model(model_dir, device)
        self._model = model
        self._device = device
        return model

    def _layer_indices(self, n_q: int) -> list[int]:
        if self.layers is None:
            return list(range(n_q))
        indices = sorted(set(int(index) for index in self.layers))
        if indices[0] < 0 or indices[-1] >= n_q:
            raise ValueError(
                f"layers must be in [0, {n_q - 1}], got {indices}"
            )
        return indices

    @torch.no_grad()
    def extract_one(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """Return [num_layers, dimension] and the number of frames."""
        model = self._ensure_model()
        waveform = load_audio(audio_path, model.sample_rate)
        codes = model.encode(waveform.unsqueeze(0).unsqueeze(0).to(self._device))
        if codes.ndim != 3 or codes.shape[1] != 1:
            raise RuntimeError(f"Unexpected code shape: {codes.shape}")
        codes = codes[:, 0, :]

        layer_indices = self._layer_indices(int(codes.shape[0]))
        quantizer_layers = model.quantizer.vq.layers
        vectors = []
        total = None  # running sum of the residual layers, i.e. the RVQ prefix
        for layer_index in layer_indices:
            quantizer = quantizer_layers[layer_index]
            frame_vectors = F.embedding(codes[layer_index], quantizer.codebook)
            frame_vectors = quantizer.project_out(frame_vectors)
            # Layer k only codes what layers 0..k-1 left over, so the decoder's
            # depth-k representation is the cumulative sum, not layer k alone.
            # With a contiguous layer list starting at 0 this is exactly q_0+..+q_k.
            total = frame_vectors if total is None else total + frame_vectors
            vectors.append(total.mean(dim=0))

        features = torch.stack(vectors).float().cpu().numpy()
        if self.normalize:
            norms = np.linalg.norm(features, axis=-1, keepdims=True)
            if np.any(norms < 1e-8):
                raise RuntimeError(f"Zero-length embedding for {audio_path}")
            features = features / norms
        return features, int(codes.shape[-1])

    def extract(
        self,
        dataset: PreparedDataset,
        output_dir: Path | None = None,
        overwrite: bool = False,
    ) -> FeatureBundle:
        output_dir = output_dir or (dataset.root / "features")
        feature_path = output_dir / "features.npz"
        if feature_path.exists() and not overwrite:
            print(f"Loading cached features from {feature_path}")
            return FeatureBundle.load(feature_path, output_dir)

        model = self._ensure_model()
        layer_indices = np.asarray(
            self._layer_indices(int(model.n_q)), dtype=np.int64
        )

        features: list[np.ndarray] = []
        for index, audio_path in enumerate(dataset.audio_paths, start=1):
            layer_features, frames = self.extract_one(audio_path)
            features.append(layer_features)
            # The MPS caching allocator fragments over hundreds of forward
            # passes and eventually runs out of memory; release it each step.
            if self._device is not None and self._device.type == "mps":
                torch.mps.empty_cache()
            print(
                f"[{index:03d}/{dataset.num_utterances:03d}] "
                f"{audio_path.name}: {frames} frames, "
                f"{layer_features.shape[0]} layers"
            )

        feature_tensor = np.stack(features)
        bundle = FeatureBundle(
            features=feature_tensor,
            speaker_ids=dataset.speaker_ids,
            genders=dataset.genders,
            utterance_ids=np.asarray(dataset.utterance_ids),
            slots=dataset.slots,
            layer_indices=layer_indices,
            split=dataset.split,
            root=output_dir,
        )
        problems = bundle.validate()
        if problems:
            raise RuntimeError("Invalid feature bundle: " + "; ".join(problems))
        bundle.save(feature_path)
        print(f"Wrote {feature_path}")
        return bundle


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract SpeechTokenizer features from a prepared dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory produced by DataPreprocessor.prepare().",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="SpeechTokenizer snapshot root; downloads if omitted.",
    )
    parser.add_argument("--device", help="Torch device (default: auto).")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Zero-based RVQ layer indices to extract (default: all).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Keep raw mean-pooled vectors.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = PreparedDataset.load(args.dataset_dir)
    extractor = SpeechTokenizerFeatureExtractor(
        model_dir=args.model_dir,
        device=args.device,
        layers=args.layers,
        normalize=not args.no_normalize,
    )
    bundle = extractor.extract(dataset, overwrite=args.overwrite)
    print(bundle.summary())


if __name__ == "__main__":
    main()
