#!/usr/bin/env python3
"""Split a transcripts file into one transcription file per audio sample.

Assumes transcripts lines are in the format:
<utt_id>\t<transcription>

And that audio files live under: <dataset_dir>/audio/<part1>/<part2>/<utt_id>.<audio_ext>
where part1 and part2 are the first two underscore-separated fields of utt_id.

Example:
  mls_polish/train/transcripts.txt contains: 6892_10350_000000\ttext...
  audio file: mls_polish/train/audio/6892/10350/6892_10350_000000.flac
  output transcription written to same dir as the audio file:
    mls_polish/train/audio/6892/10350/6892_10350_000000.txt

This script supports a dry-run mode and basic safety options.
"""

import argparse
import io
import os
import sys
import glob


def build_audio_index(dataset_dir, audio_subdir):
    """Recursively scan dataset_dir/<audio_subdir> (or dataset_dir if subdir missing)
    and build a mapping from utterance id (filename without extension) -> file path.
    If multiple files share the same utt id, the first encountered path is kept.
    """
    index = {}
    root = os.path.join(dataset_dir, audio_subdir)
    if not os.path.isdir(root):
        # fallback to dataset_dir root
        root = dataset_dir

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            stem, _ = os.path.splitext(fn)
            if stem in index:
                # already indexed; skip duplicates to keep first hit
                continue
            full = os.path.join(dirpath, fn)
            index[stem] = full

    return index


def process(transcripts_file, dataset_dir, audio_subdir, audio_ext, out_ext, dry_run, overwrite, skip_existing, limit=None):
    total = 0
    written = 0
    missing = 0
    errors = 0

    # build an index of audio files for fast recursive lookup
    audio_index = build_audio_index(dataset_dir, audio_subdir)
    if not audio_index:
        print(f"Warning: no audio files found under {os.path.join(dataset_dir, audio_subdir)}")

    with io.open(transcripts_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and total >= limit:
                break
            total += 1
            line = line.strip()
            if not line:
                continue
            # try tab split first, then whitespace
            if "\t" in line:
                utt_id, text = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    utt_id, text = parts
                else:
                    print(f"Skipping malformed line {total}: {line!r}")
                    errors += 1
                    continue

            utt_id = utt_id.strip()
            text = text.strip()

            # lookup in pre-built index (fast, recursive)
            audio_path = audio_index.get(utt_id)
            if not audio_path and audio_ext:
                # try with extension suffix in case stems differ
                candidate = utt_id + audio_ext
                audio_path = audio_index.get(os.path.splitext(candidate)[0])
            if not audio_path:
                missing += 1
                print(f"[MISSING] {utt_id} -> audio not found under {dataset_dir}/{audio_subdir}")
                continue

            out_dir = os.path.dirname(audio_path)
            out_file = os.path.join(out_dir, utt_id + out_ext)

            if os.path.exists(out_file):
                if skip_existing:
                    print(f"[SKIP] {out_file} already exists")
                    continue
                if not overwrite:
                    print(f"[EXISTS] {out_file} (use --overwrite to replace)")
                    continue

            print(f"[WRITE] {out_file}")
            if dry_run:
                written += 1
                continue

            try:
                # write UTF-8 text file
                with io.open(out_file, "w", encoding="utf-8") as ofh:
                    ofh.write(text + "\n")
                written += 1
            except Exception as e:
                errors += 1
                print(f"[ERROR] writing {out_file}: {e}")

    print("\nSummary:")
    print(f"  processed lines: {total}")
    print(f"  planned/written (dry-run or actual): {written}")
    print(f"  missing audio files: {missing}")
    print(f"  errors: {errors}")


def main():
    p = argparse.ArgumentParser(description="Split transcripts into per-utterance files next to audio")
    p.add_argument("--transcripts", "-t", required=True, help="Path to transcripts.txt")
    p.add_argument("--dataset-dir", "-d", required=True, help="Path to dataset directory containing `audio/`")
    p.add_argument("--audio-subdir", default="audio", help="Name of audio subdirectory under dataset-dir (default: audio)")
    p.add_argument("--audio-ext", default=".flac", help="Audio extension to look for (default: .flac)")
    p.add_argument("--out-ext", default=".lab", help="Output transcription extension (default: .txt)")
    p.add_argument("--dry-run", action="store_true", help="Don't write files; only print actions")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing transcription files")
    p.add_argument("--skip-existing", action="store_true", help="Skip writing if output file exists")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N lines (useful for testing)")

    args = p.parse_args()

    if not os.path.exists(args.transcripts):
        print(f"transcripts file not found: {args.transcripts}")
        sys.exit(2)

    if not os.path.isdir(args.dataset_dir):
        print(f"dataset dir not found: {args.dataset_dir}")
        sys.exit(2)

    process(args.transcripts, args.dataset_dir, args.audio_subdir, args.audio_ext, args.out_ext, args.dry_run, args.overwrite, args.skip_existing, args.limit)


if __name__ == "__main__":
    main()
