#!/usr/bin/env python3
"""Download and prepare a class-balanced subset of any LibriTTS split.

This module contains only data preparation: speaker selection, audio download,
transcripts, and metadata. SpeechTokenizer feature extraction is intentionally
kept in ``speechtokenizer_features.py``, and cross-validation splits are kept
outside the class.

Example:
    pre = DataPreprocessor(
        split="dev.clean",
        num_speakers=36,
        utterances_per_speaker=20,
    )
    dataset = pre.prepare()
    print(dataset.summary())
"""

from __future__ import annotations

import json
import random
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

BASE_DATASET = "mythicinfinity/libritts_r"
TAGS_DATASET = "ylacombe/libritts_r_tags"
MODELSCOPE_DATASET_ID = "pengzhendong/libritts_r"
MODELSCOPE_DATASET_URL = (
    f"https://www.modelscope.cn/datasets/{MODELSCOPE_DATASET_ID}"
)
DATASETS_SERVER = "https://datasets-server.huggingface.co"
USER_AGENT = "ssl-probing-data-preprocessor/1.0"

SPLIT_TO_TAGS_CONFIG = {
    "dev.clean": "clean",
    "test.clean": "clean",
    "train.clean.100": "clean",
    "train.clean.360": "clean",
    "dev.other": "other",
    "test.other": "other",
    "train.other.500": "other",
}


@dataclass(frozen=True)
class SpeakerPlan:
    speaker_id: str
    gender: str
    row_indices: tuple[int, ...]
    utterance_ids: tuple[str, ...]


@dataclass
class PreparedDataset:
    split: str
    root: Path
    audio_paths: list[Path]
    utterance_ids: list[str]
    speaker_ids: np.ndarray
    genders: np.ndarray
    slots: np.ndarray
    metadata: list[dict]

    @property
    def num_utterances(self) -> int:
        return len(self.utterance_ids)

    @property
    def num_speakers(self) -> int:
        return len(set(self.speaker_ids.tolist()))

    def summary(self) -> dict[str, object]:
        speaker_gender = {}
        for speaker, gender in zip(self.speaker_ids.tolist(), self.genders.tolist()):
            speaker_gender[str(speaker)] = str(gender)
        speaker_counts = Counter(self.speaker_ids.tolist())
        per_speaker = sorted(speaker_counts.values())
        return {
            "split": self.split,
            "num_speakers": self.num_speakers,
            "num_utterances": self.num_utterances,
            "speaker_gender_counts": dict(Counter(speaker_gender.values())),
            "utterance_gender_counts": dict(Counter(self.genders.tolist())),
            "utterances_per_speaker_min": int(min(per_speaker)),
            "utterances_per_speaker_max": int(max(per_speaker)),
            "root": str(self.root),
        }

    def validate(self) -> list[str]:
        problems: list[str] = []
        if len(set(self.utterance_ids)) != self.num_utterances:
            problems.append("Duplicate utterance IDs")
        for path in self.audio_paths:
            if not path.exists():
                problems.append(f"Missing audio: {path}")
        speaker_counts = Counter(self.speaker_ids.tolist())
        if len(set(speaker_counts.values())) != 1:
            problems.append("Speakers do not all have the same utterance count")
        for speaker, count in speaker_counts.items():
            expected = set(range(count))
            actual = set(
                self.slots[self.speaker_ids == speaker].tolist()
            )
            if actual != expected:
                problems.append(f"Bad slots for speaker {speaker}: {sorted(actual)}")
        return problems

    @classmethod
    def load(cls, root: Path) -> "PreparedDataset":
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata.json under {root}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        entries = payload["files"]
        return cls(
            split=payload["split"],
            root=root,
            audio_paths=[root / entry["file"] for entry in entries],
            utterance_ids=[entry["id"] for entry in entries],
            speaker_ids=np.asarray(
                [entry["speaker_id"] for entry in entries]
            ),
            genders=np.asarray([entry["gender"] for entry in entries]),
            slots=np.asarray([entry["slot"] for entry in entries], dtype=np.int64),
            metadata=entries,
        )


def _http_json(url: str, retries: int = 4, timeout: float = 60.0) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            last_error = error
            if attempt + 1 < retries:
                delay = (
                    20.0 * (attempt + 1)
                    if error.code == 429
                    else 3.0 * (attempt + 1)
                )
                time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}\n{last_error}")


def _download_url(url: str, destination: Path, retries: int = 3) -> None:
    temporary = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            temporary.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}\n{last_error}")


def _read_wav_info(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as wav_file:
        return (
            wav_file.getframerate(),
            wav_file.getnchannels(),
            wav_file.getnframes(),
        )


def _contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    sorted_indices = sorted(indices)
    ranges: list[tuple[int, int]] = []
    start = previous = sorted_indices[0]
    for index in sorted_indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append((start, previous))
        start = previous = index
    ranges.append((start, previous))
    return ranges


class DataPreprocessor:
    """Select and download a balanced subset of any LibriTTS-R split."""

    def __init__(
        self,
        split: str = "dev.clean",
        num_speakers: int = 36,
        utterances_per_speaker: int = 20,
        balance_gender: bool = True,
        seed: int = 0,
        cache_dir: Path = Path("data/libritts_cache"),
        output_dir: Path = Path("data/libritts_preprocessed"),
        overwrite: bool = False,
        strict: bool = True,
    ) -> None:
        if split not in SPLIT_TO_TAGS_CONFIG:
            raise ValueError(
                f"Unsupported split {split!r}; choose from "
                f"{sorted(SPLIT_TO_TAGS_CONFIG)}"
            )
        if num_speakers < 2:
            raise ValueError("num_speakers must be at least 2")
        if utterances_per_speaker < 2:
            raise ValueError("utterances_per_speaker must be at least 2")

        self.split = split
        self.num_speakers = num_speakers
        self.utterances_per_speaker = utterances_per_speaker
        self.balance_gender = balance_gender
        self.seed = seed
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.strict = strict

    @staticmethod
    def available_splits() -> list[str]:
        return list(SPLIT_TO_TAGS_CONFIG)

    def _tags_path(self) -> Path:
        return self.cache_dir / f"{self.split}.parquet"

    def _ensure_tags(self) -> Path:
        path = self._tags_path()
        if path.exists() and not self.overwrite:
            return path
        config = SPLIT_TO_TAGS_CONFIG[self.split]
        url = (
            f"https://huggingface.co/datasets/{TAGS_DATASET}/resolve/main/"
            f"{config}/{self.split}-00000-of-00001.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading tags for {self.split} to {path}")
        _download_url(url, path)
        return path

    @property
    def modelscope_url(self) -> str:
        return MODELSCOPE_DATASET_URL

    def modelscope_shards(self) -> list[dict[str, object]]:
        """List the full-split Parquet shards mirrored on ModelScope."""
        root = f"data/{self.split}"
        page_number = 1
        page_size = 200
        shards: list[dict[str, object]] = []
        while True:
            query = urllib.parse.urlencode(
                {
                    "Revision": "master",
                    "Recursive": "True",
                    "PageNumber": page_number,
                    "PageSize": page_size,
                    "Root": root,
                }
            )
            payload = _http_json(
                f"https://www.modelscope.cn/api/v1/datasets/"
                f"{MODELSCOPE_DATASET_ID}/repo/tree?{query}"
            )
            files = payload.get("Data", {}).get("Files", [])
            if not files:
                break
            for entry in files:
                if entry.get("Type") != "blob":
                    continue
                path = str(entry["Path"])
                shards.append(
                    {
                        "path": path,
                        "size": int(entry.get("Size", 0)),
                        "url": (
                            f"https://www.modelscope.cn/api/v1/datasets/"
                            f"{MODELSCOPE_DATASET_ID}/repo?Revision=master"
                            f"&FilePath={urllib.parse.quote(path, safe='')}"
                        ),
                    }
                )
            if len(files) < page_size:
                break
            page_number += 1
        shards.sort(key=lambda item: str(item["path"]))
        return shards

    def download_modelscope_backup(
        self,
        max_shards: int | None = None,
    ) -> list[Path]:
        """Download the split's full ModelScope Parquet shards as a backup.

        ModelScope mirrors full split shards rather than individual rows, so
        this is intentionally a whole-split fallback.
        """
        shards = self.modelscope_shards()
        if max_shards is not None:
            shards = shards[:max_shards]
        if not shards:
            raise RuntimeError(
                f"No ModelScope shards found for {self.split} at "
                f"{self.modelscope_url}"
            )

        total_bytes = sum(int(shard["size"]) for shard in shards)
        print(
            f"ModelScope backup for {self.split}: {len(shards)} shard(s), "
            f"{total_bytes / 1e9:.2f} GB from {self.modelscope_url}"
        )
        backup_dir = self.cache_dir / "modelscope" / self.split
        backup_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index, shard in enumerate(shards, start=1):
            destination = backup_dir / Path(str(shard["path"])).name
            if not destination.exists() or self.overwrite:
                print(
                    f"[{index}/{len(shards)}] {destination.name} "
                    f"({int(shard['size']) / 1e6:.1f} MB)"
                )
                _download_url(str(shard["url"]), destination)
            else:
                print(f"[{index}/{len(shards)}] {destination.name} already exists")
            paths.append(destination)
        return paths

    def _load_tags(
        self,
    ) -> tuple[dict[str, list[int]], dict[str, str], list[str]]:
        path = self._ensure_tags()
        table = pq.read_table(path, columns=["speaker_id", "gender", "id"])
        speaker_ids = table.column("speaker_id").to_pylist()
        genders = table.column("gender").to_pylist()
        utterance_ids = [str(value) for value in table.column("id").to_pylist()]

        rows_by_speaker: dict[str, list[int]] = defaultdict(list)
        gender_by_speaker: dict[str, str] = {}
        for row_index, speaker_id in enumerate(speaker_ids):
            speaker = str(speaker_id)
            rows_by_speaker[speaker].append(row_index)
            gender = str(genders[row_index]).lower()
            previous = gender_by_speaker.setdefault(speaker, gender)
            if previous != gender:
                raise RuntimeError(f"Inconsistent gender for speaker {speaker}")
        return dict(rows_by_speaker), gender_by_speaker, utterance_ids

    def speaker_counts(self) -> dict[str, int]:
        rows_by_speaker, _, _ = self._load_tags()
        return {
            speaker: len(rows)
            for speaker, rows in sorted(rows_by_speaker.items())
        }

    def summary(self) -> dict[str, object]:
        counts = self.speaker_counts()
        values = sorted(counts.values())
        return {
            "split": self.split,
            "num_speakers": len(counts),
            "num_utterances": int(sum(values)),
            "min_utterances_per_speaker": int(values[0]),
            "median_utterances_per_speaker": float(np.median(values)),
            "max_utterances_per_speaker": int(values[-1]),
            "requested_speakers": self.num_speakers,
            "requested_utterances_per_speaker": self.utterances_per_speaker,
        }

    def plan(self) -> list[SpeakerPlan]:
        rows_by_speaker, gender_by_speaker, utterance_ids = self._load_tags()
        eligible = [
            speaker
            for speaker, rows in rows_by_speaker.items()
            if len(rows) >= self.utterances_per_speaker
        ]
        if not eligible:
            raise RuntimeError(
                f"No speaker in {self.split} has "
                f"{self.utterances_per_speaker} utterances"
            )

        rng = random.Random(self.seed)
        target_female = self.num_speakers // 2
        target_male = self.num_speakers - target_female

        if self.balance_gender:
            female = [
                speaker
                for speaker in eligible
                if gender_by_speaker[speaker] == "female"
            ]
            male = [
                speaker
                for speaker in eligible
                if gender_by_speaker[speaker] == "male"
            ]
            rng.shuffle(female)
            rng.shuffle(male)
            available_female = min(target_female, len(female))
            available_male = min(target_male, len(male))
            if (
                available_female < target_female
                or available_male < target_male
            ) and self.strict:
                raise RuntimeError(
                    f"{self.split} has only {len(female)} female and "
                    f"{len(male)} male speakers with "
                    f">={self.utterances_per_speaker} utterances; "
                    f"requested {target_female}F/{target_male}M. "
                    "Lower num_speakers, lower utterances_per_speaker, "
                    "or use strict=False."
                )
            selected = female[:available_female] + male[:available_male]
            if len(selected) < self.num_speakers:
                print(
                    f"Warning: only {len(selected)} balanced speakers are "
                    f"available; using {len(selected)} instead of "
                    f"{self.num_speakers}"
                )
        else:
            rng.shuffle(eligible)
            selected = eligible[: self.num_speakers]
            if len(selected) < self.num_speakers and self.strict:
                raise RuntimeError(
                    f"{self.split} has only {len(eligible)} speakers with "
                    f">={self.utterances_per_speaker} utterances; requested "
                    f"{self.num_speakers}. Lower num_speakers, lower "
                    "utterances_per_speaker, or use strict=False."
                )
            if len(selected) < self.num_speakers:
                print(
                    f"Warning: only {len(selected)} speakers are available; "
                    f"using {len(selected)} instead of {self.num_speakers}"
                )

        selected.sort(key=lambda speaker: (len(speaker), speaker))
        plans: list[SpeakerPlan] = []
        for speaker in selected:
            indices = sorted(rows_by_speaker[speaker])
            runs = [
                (start, end)
                for start, end in _contiguous_ranges(indices)
                if end - start + 1 >= self.utterances_per_speaker
            ]
            if runs:
                start, end = rng.choice(runs)
                first = rng.randint(
                    start, end - self.utterances_per_speaker + 1
                )
                chosen = list(
                    range(first, first + self.utterances_per_speaker)
                )
            else:
                chosen = sorted(
                    rng.sample(indices, self.utterances_per_speaker)
                )

            plans.append(
                SpeakerPlan(
                    speaker_id=speaker,
                    gender=gender_by_speaker[speaker],
                    row_indices=tuple(chosen),
                    utterance_ids=tuple(
                        utterance_ids[index] for index in chosen
                    ),
                )
            )
        return plans

    def _fetch_rows(
        self,
        offset: int,
        length: int,
    ) -> list[dict]:
        query = urllib.parse.urlencode(
            {
                "dataset": BASE_DATASET,
                "config": "all",
                "split": self.split,
                "offset": offset,
                "length": length,
            }
        )
        payload = _http_json(f"{DATASETS_SERVER}/rows?{query}")
        rows = payload.get("rows", [])
        if not rows:
            raise RuntimeError(f"No rows at offset {offset}")
        return rows

    def _fetch_base_rows(self, indices: list[int]) -> list[dict]:
        by_index: dict[int, dict] = {}
        for start, end in _contiguous_ranges(indices):
            rows = self._fetch_rows(start, end - start + 1)
            for item in rows:
                by_index[int(item["row_idx"])] = item["row"]
        missing = [index for index in indices if index not in by_index]
        if missing:
            raise RuntimeError(f"Missing base rows: {missing[:5]}")
        return [by_index[index] for index in indices]

    def prepare(self) -> PreparedDataset:
        plans = self.plan()
        dataset_dir = (
            self.output_dir
            / f"{self.split}_{len(plans)}speakers_"
            f"{self.utterances_per_speaker}utts"
        )
        metadata_path = dataset_dir / "metadata.json"

        if metadata_path.exists() and not self.overwrite:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                payload.get("split") == self.split
                and payload.get("num_speakers") == len(plans)
                and payload.get("utterances_per_speaker")
                == self.utterances_per_speaker
                and payload.get("seed") == self.seed
            ):
                return PreparedDataset.load(dataset_dir)

        audio_dir = dataset_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict] = []
        total = len(plans) * self.utterances_per_speaker
        done = 0

        for plan in plans:
            rows = self._fetch_base_rows(list(plan.row_indices))
            if len(rows) != len(plan.row_indices):
                raise RuntimeError(f"Row fetch count mismatch for {plan.speaker_id}")
            for slot, (row_index, utterance_id, row) in enumerate(
                zip(plan.row_indices, plan.utterance_ids, rows)
            ):
                if str(row["id"]) != utterance_id:
                    raise RuntimeError(
                        f"Row alignment mismatch: {row['id']} != {utterance_id}"
                    )
                if str(row["speaker_id"]) != plan.speaker_id:
                    raise RuntimeError(
                        f"Speaker mismatch: {row['speaker_id']} != "
                        f"{plan.speaker_id}"
                    )

                destination = audio_dir / f"{utterance_id}.wav"
                if self.overwrite or not destination.exists():
                    audio = row.get("audio")
                    if not isinstance(audio, list) or not audio:
                        raise RuntimeError(f"No audio for {utterance_id}")
                    _download_url(str(audio[0]["src"]), destination)

                transcript = str(
                    row.get("text_normalized") or row.get("text_original") or ""
                )
                (destination.with_suffix(".txt")).write_text(
                    transcript.strip() + "\n", encoding="utf-8"
                )

                sample_rate, channels, frames = _read_wav_info(destination)
                entries.append(
                    {
                        "file": str(destination.relative_to(dataset_dir)),
                        "id": utterance_id,
                        "offset": int(row_index),
                        "speaker_id": plan.speaker_id,
                        "gender": plan.gender,
                        "slot": slot,
                        "chapter_id": str(row.get("chapter_id", "")),
                        "text": transcript,
                        "duration_s": round(frames / sample_rate, 4),
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "source_path": str(row.get("path", "")),
                    }
                )
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"[{done:04d}/{total:04d}] {utterance_id}")

        payload = {
            "dataset": BASE_DATASET,
            "split": self.split,
            "seed": self.seed,
            "balance_gender": self.balance_gender,
            "num_speakers": len(plans),
            "utterances_per_speaker": self.utterances_per_speaker,
            "num_utterances": len(entries),
            "files": entries,
        }
        metadata_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

        return PreparedDataset(
            split=self.split,
            root=dataset_dir,
            audio_paths=[dataset_dir / entry["file"] for entry in entries],
            utterance_ids=[entry["id"] for entry in entries],
            speaker_ids=np.asarray(
                [entry["speaker_id"] for entry in entries]
            ),
            genders=np.asarray([entry["gender"] for entry in entries]),
            slots=np.asarray([entry["slot"] for entry in entries], dtype=np.int64),
            metadata=entries,
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare a balanced subset of any LibriTTS split."
    )
    parser.add_argument("--split", default="dev.clean")
    parser.add_argument("--num-speakers", type=int, default=36)
    parser.add_argument("--utterances-per-speaker", type=int, default=20)
    parser.add_argument("--no-balance-gender", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/libritts_cache"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/libritts_preprocessed")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument(
        "--download-modelscope-backup",
        action="store_true",
        help="Download the full split Parquet shards from ModelScope and exit.",
    )
    parser.add_argument(
        "--modelscope-max-shards",
        type=int,
        help="Limit the number of ModelScope shards downloaded.",
    )
    args = parser.parse_args()
    
    
if __name__ == "__main__":

    # 1. Initialize the preprocessor
    preprocessor = DataPreprocessor(
        split="dev.clean",
        num_speakers=36,
        utterances_per_speaker=20,
        balance_gender=True,
        seed=42,
        cache_dir=Path("data/libritts_cache"),
        output_dir=Path("data/libritts_preprocessed"),
    )

    # 2. Prepare the dataset (downloads audio, transcripts, and generates metadata.json)
    dataset = preprocessor.prepare()

    # 3. Print dataset summary statistics
    print("Summary:", dataset.summary())

    # 4. Validate dataset file existence and alignment
    problems = dataset.validate()
    if problems:
        print("Validation issues found:", problems)
    else:
        print("Dataset validation passed successfully!")

    # 5. Access individual tensors and file paths
    print(f"Total utterances: {dataset.num_utterances}")
    print(f"Total speakers: {dataset.num_speakers}")
    print(f"First audio path: {dataset.audio_paths[0]}")
    print(f"Speaker IDs shape: {dataset.speaker_ids.shape}")