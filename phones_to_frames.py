"""Turn MFA TextGrid alignments into frame-level phone labels matching RVQ features.

The SpeechTokenizer encoder runs at 16 kHz with a 320-sample hop, i.e. 50 frames
per second (20 ms per frame) -- NOT 25 fps. Labels must be rasterised at the
same rate as the features, or every phone covers half the frames it should.
Every phone interval from the aligner is mapped onto that grid:

    start = floor(onset  * fps)
    end   = ceil (offset * fps)      (end exclusive)

Alignment quality is checked here, not assumed: phones shorter than 20 ms or
longer than 400 ms are counted per file, and median phone durations are printed
so obvious misalignments show up before they reach a probe.

    .venv/bin/python phones_to_frames.py --textgrids aligned/
"""

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data" / "libritts_preprocessed" / "dev.clean_36speakers_20utts"
FPS = 50.0  # SpeechTokenizer runs at 16 kHz / 320 hop = 50 frames/s
MIN_PHONE_S = 0.02
MAX_PHONE_S = 0.40
SILENCE = {"", "sil", "sp", "spn", "<eps>"}


def read_grid_text(path: Path) -> str:
    """Read a TextGrid, honouring the UTF-16 BOM Praat writes on some platforms."""
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def parse_textgrid(path: Path) -> list[tuple[float, float, str]]:
    """Return (onset, offset, label) for the ``phones`` tier, or [] if absent.

    MFA writes TextGrids in two layouts (older inverted ones put ``intervals``
    before ``xmin``; current ones are standard), so intervals are collected as
    pending xmin/xmax pairs and closed when a ``text`` value appears.
    """
    text = unicodedata.normalize("NFC", read_grid_text(path))
    intervals = []
    for tier in text.split("item [")[1:]:
        name = None
        for line in tier.splitlines():
            line = line.strip()
            if line.startswith("name ="):
                name = line.split("=", 1)[1].strip().strip('"')
                break
        if name != "phones":
            continue
        pending = {}
        for line in tier.splitlines():
            line = line.strip()
            if "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            if key in ("xmin", "xmax"):
                pending[key] = float(value)
            elif key == "text" and {"xmin", "xmax"} <= pending.keys():
                intervals.append((pending["xmin"], pending["xmax"], value.strip('"').strip().lower()))
                pending = {}
    intervals.sort()
    return intervals


def parse_tier_lab(path: Path) -> list[tuple[float, float, str]]:
    """Parse Praat long TextGrid files (``--format lab``), used by our MFA tests."""
    intervals = []
    for line in read_grid_text(path).splitlines():
        match = re.match(r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(\S.*?)\s*$", line)
        if match:
            intervals.append((float(match.group(1)), float(match.group(2)), match.group(3).lower()))
    return sorted(intervals)


def load_alignment(path: Path) -> list[tuple[float, float, str]]:
    if path.suffix.lower() == ".lab":
        return parse_tier_lab(path)
    return parse_textgrid(path)


def canonical(phone: str, keep_stress: bool) -> str:
    return phone if keep_stress else phone.rstrip("012")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textgrids", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--keep-stress", action="store_true",
                        help="keep ARPAbet stress digits (AH0/AH1/AH2 as separate classes)")
    args = parser.parse_args()
    out = args.data_dir / "features" / "phones.npz"

    features = np.load(args.data_dir / "features" / "features.npz", allow_pickle=False)
    utterance_ids = [str(u) for u in features["utterance_ids"]]
    meta = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    duration = {e["id"]: float(e["duration_s"]) for e in meta["files"]}

    grids = [g for g in sorted(list(args.textgrids.glob("*.TextGrid")) + list(args.textgrids.glob("*.lab")))
             if g.stem in duration]
    if not grids:
        raise SystemExit(f"no .TextGrid files under {args.textgrids}; run `mfa align` first")

    # Pass 1: count phones so ids are ordered by frequency (id 0 = most common).
    parsed = {}
    counts = Counter()
    for grid in grids:
        intervals = load_alignment(grid)
        parsed[grid.stem] = intervals
        for _, _, phone in intervals:
            phone = canonical(phone, args.keep_stress)
            if phone not in SILENCE:
                counts[phone] += 1
    phones = sorted(counts, key=lambda p: (-counts[p], p))
    phone_to_id = {p: i for i, p in enumerate(phones)}

    # Pass 2: rasterise onto the encoder's frame grid and collect quality stats.
    rows, durations = {}, defaultdict(list)
    short, long, unaligned = Counter(), Counter(), []
    for utt, intervals in parsed.items():
        if not intervals:
            unaligned.append(utt)
            continue
        n_frames = int(round(duration[utt] * args.fps))
        row = np.full(n_frames, -1, dtype=np.int16)  # -1 = silence/pause/uncovered
        for onset, offset, phone in intervals:
            phone = canonical(phone, args.keep_stress)
            if phone in SILENCE:
                continue
            span = offset - onset
            durations[phone].append(span)
            if span < MIN_PHONE_S:
                short[utt] += 1
            elif span > MAX_PHONE_S:
                long[utt] += 1
            start = max(0, int(np.floor(onset * args.fps)))
            end = min(n_frames, int(np.ceil(offset * args.fps)))
            row[start:end] = phone_to_id[phone]
        rows[utt] = row

    # --- alignment quality report -----------------------------------------
    total = sum(counts.values())
    print(f"aligned files: {len(rows)}/{len(duration)}")
    if unaligned:
        print(f"textgrids without a usable 'phones' tier: {len(unaligned)}")
    print(f"phones <{int(MIN_PHONE_S*1000)} ms: {sum(short.values())} ({sum(short.values())/max(total,1):.1%})"
          f" | files with >=5 of them: {sum(1 for n in short.values() if n >= 5)}")
    print(f"phones >{int(MAX_PHONE_S*1000)} ms: {sum(long.values())} ({sum(long.values())/max(total,1):.1%})")
    for utt, n in sorted(short.items(), key=lambda kv: -kv[1])[:5]:
        if n >= 5:
            print(f"  check alignment: {utt} ({n} very short phones)")

    print(f"\ndistinct phones: {len(phones)} | total phone tokens: {total}")
    print("phone  count  median ms")
    for phone in phones[:12]:
        print(f"  {phone:4s} {counts[phone]:5d}   {np.median(durations[phone])*1000:6.1f}")
    suspect = [p for p in phones if np.median(durations[p]) < 2 * MIN_PHONE_S]
    if suspect:
        print(f"suspect phones (median < {2*MIN_PHONE_S*1000:.0f} ms): {suspect}")

    # --- save --------------------------------------------------------------
    order = [u for u in utterance_ids if u in rows]
    max_len = max((len(rows[u]) for u in order), default=0)
    table = np.full((len(order), max_len), -1, dtype=np.int16)
    lengths = np.zeros(len(order), dtype=np.int32)
    for i, utt in enumerate(order):
        table[i, : len(rows[utt])] = rows[utt]
        lengths[i] = len(rows[utt])

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        labels=table,
        lengths=lengths,   # per-clip frame count; padding past `lengths` is not real audio
        utterance_ids=np.asarray(order),
        phone_names=np.asarray(phones),
        fps=np.asarray(args.fps),
    )
    if table.size == 0:
        raise SystemExit("no phone intervals parsed - check the TextGrid format")
    # Coverage must exclude the -1 padding: a 0.6 s clip padded to 582 columns
    # would otherwise look 2% labelled when its own frames are mostly phones.
    valid = np.zeros_like(table, dtype=bool)
    for i, n in enumerate(lengths):
        valid[i, :n] = True
    per_clip = np.array([(table[i, :n] >= 0).mean() for i, n in enumerate(lengths) if n])
    coverage = float((table[valid] >= 0).mean())
    print(f"\nframes labelled by a phone: {coverage:.1%} of real audio frames")
    if per_clip.size:
        print(f"per-clip coverage: min {per_clip.min():.1%} | median {np.median(per_clip):.1%} | max {per_clip.max():.1%}")
    print(f"wrote {out}  shape {table.shape}")


if __name__ == "__main__":
    main()