#!/usr/bin/env python3
"""
Create .lab files for LibriSpeech-style dataset directories.

For each directory under the provided root, this script looks for files
ending with `.trans.txt`. Each line in a `.trans.txt` is expected to be
of the form:

    <utterance-id> <transcript text...>

For each such line the script will write a file named `<utterance-id>.lab`
in the same directory containing the transcript text.

Usage:
    python3 create_lab_files.py --root /abs/path/to/LibriSpeech/train-clean-100

By default the script will overwrite existing `.lab` files. Use
`--skip-existing` to avoid overwriting.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple


def process_trans_file(trans_path: str, skip_existing: bool = False) -> Tuple[int, int]:
    """Read a .trans.txt file and create .lab files next to it.

    Returns (created, skipped)
    """
    created = 0
    skipped = 0
    dirpath = os.path.dirname(trans_path)
    try:
        with open(trans_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                # Split only on first whitespace to preserve transcript spacing
                parts = ln.split(None, 1)
                if not parts:
                    continue
                if len(parts) == 1:
                    uttid = parts[0]
                    transcript = ""
                else:
                    uttid, transcript = parts

                lab_path = os.path.join(dirpath, uttid + ".lab")
                if skip_existing and os.path.exists(lab_path):
                    skipped += 1
                    continue

                # Write transcript (single line). Overwrite by default.
                with open(lab_path, "w", encoding="utf-8") as out:
                    out.write(transcript + "\n")
                created += 1
    except Exception:
        print(f"Failed to process '{trans_path}':", file=sys.stderr)
        raise

    return created, skipped


def walk_and_create(root: str, skip_existing: bool = False) -> Tuple[int, int, int]:
    """Walk directory tree rooted at `root` and process all `*.trans.txt` files.

    Returns (num_trans_files, total_created, total_skipped)
    """
    num_trans = 0
    total_created = 0
    total_skipped = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".trans.txt"):
                num_trans += 1
                trans_path = os.path.join(dirpath, fn)
                created, skipped = process_trans_file(trans_path, skip_existing=skip_existing)
                total_created += created
                total_skipped += skipped

    return num_trans, total_created, total_skipped


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate .lab files from LibriSpeech .trans.txt files")
    p.add_argument("--root", required=False,
                   default=os.path.join(os.path.dirname(__file__), "LibriSpeech", "train-clean-100"),
                   help="Root directory to walk (default: Koushik/LibriSpeech/train-clean-100)")
    p.add_argument("--skip-existing", action="store_true", help="Do not overwrite existing .lab files")
    args = p.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Root directory does not exist or is not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    print(f"Scanning for .trans.txt under: {root}")
    num_trans, total_created, total_skipped = walk_and_create(root, skip_existing=args.skip_existing)

    print("Done.")
    print(f"  .trans.txt files processed: {num_trans}")
    print(f"  .lab files created: {total_created}")
    print(f"  .lab files skipped (skip-existing=True): {total_skipped}")


if __name__ == "__main__":
    main()
