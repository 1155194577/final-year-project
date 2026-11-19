"""Build an MFA corpus from the prepared LibriTTS subset.

MFA expects ``<corpus>/<name>.wav`` next to ``<corpus>/<name>.lab``. This script
pairs the preprocessed WAVs with their transcripts, normalises the text for the
``english_us_arpa`` dictionary (lowercase, straight quotes, no stray Unicode),
and reports anything that would make the aligner guess.

    .venv/bin/python make_mfa_corpus.py                 # writes mfa_corpus/
    .venv/bin/python make_mfa_corpus.py --check-text    # inspect text/audio offset risk
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "libritts_preprocessed" / "dev.clean_36speakers_20utts"
CORPUS_DIR = Path(__file__).parent / "mfa_corpus"

# Characters that appear in the source books and break the arpa dictionary.
QUOTE_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": "'", "\u201d": "'",
    "\u00ab": "'", "\u00bb": "'", "\u2014": " ", "\u2013": " ",
    "\u2026": " ", "\u00e9": "e", "\u00e0": "a", "\u00e8": "e",
}
KEEP = re.compile(r"[a-z'\- ]+")
MAX_DURATION_S = 30.0
MIN_DURATION_S = 0.5


def normalise(text: str) -> str:
    """NFC, straight quotes, lowercase, ASCII letters/apostrophe/hyphen only."""
    text = unicodedata.normalize("NFC", text)
    for src, dst in QUOTE_MAP.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    # Keep only the characters the arpa dictionary knows; everything else
    # (punctuation, digits) becomes a space so word boundaries survive.
    text = "".join(ch if KEEP.fullmatch(ch) else " " for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def check_text(entries):
    """Report how likely the transcript is to start before the audio does.

    Utterances were cut from mid-paragraph LibriTTS audio, so the chunk text may
    open with words that are not in the clip. A high rate of clips starting with
    a conjunction is the tell.
    """
    firsts = Counter(e["text"].split()[0].lower() for e in entries if e["text"].split())
    risky = sum(n for w, n in firsts.items() if w in {"and", "but", "then", "so", "for"})
    capitalised = sum(1 for e in entries if e["text"].strip()[:1].isupper())
    print(f"clips starting with a conjunction: {risky}/{len(entries)}")
    print(f"clips starting capitalised:        {capitalised}/{len(entries)}")
    print(f"most common first words:           {firsts.most_common(5)}")
    print(
        "\nIf 'conjunction' is large, parts of the leading text may not be spoken "
        "in the clip; MFA will still force an alignment. Verify one clip's first "
        "phone starts near 0 s before trusting the labels."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=CORPUS_DIR)
    parser.add_argument("--check-text", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    meta = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    entries = meta["files"]

    if args.check_text:
        check_text(entries)

    args.out.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, []
    empty_after_norm = 0
    for entry in entries:
        wav = args.data_dir / entry["file"]
        if not wav.exists():
            skipped.append((entry["id"], "missing wav"))
            continue
        duration = float(entry.get("duration_s", 0.0))
        if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
            skipped.append((entry["id"], f"duration {duration:.2f}s"))
            continue
        text = normalise(entry["text"])
        if not text:
            empty_after_norm += 1
            skipped.append((entry["id"], "empty after normalisation"))
            continue
        if not args.dry_run:
            # Symlink the audio to avoid a second copy of 190 MB.
            link = args.out / f"{entry['id']}.wav"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(wav.resolve())
            (args.out / f"{entry['id']}.lab").write_text(text + "\n", encoding="utf-8")
        written += 1

    print(f"\nwrote {written} wav/lab pairs to {args.out}")
    if empty_after_norm:
        print(f"empty after normalisation: {empty_after_norm}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for utt, reason in skipped[:20]:
            print(f"  {utt}: {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")


if __name__ == "__main__":
    main()