#!/usr/bin/env python3
"""
Copy audio files and transcripts listed in a handles file into a
<language>_low_resource directory.

Behavior matches the existing bash script:
- Determine repository root by looking for an ancestor named `mls_*`.
- Derive language from that folder name (mls_<language>). If not found,
  infer from provided dest or repo folder name.
- Default handles file: <repo_root>/train/limited_supervision/9hr/handles.txt
- Default dest: <repo_root>/<language>_low_resource
- Probe audio extensions: .flac, .wav, .mp3
- Copy transcripts: prefer .lab, fall back to .txt; write dest transcripts as .lab
- No-clobber: do not overwrite existing files (log SKIPPED-EXISTS)
- Dry-run support: --dry-run / -n
- Logging to DEST/copy_log.tsv (TSV with timestamp, status, src, dest)

Usage:
  python3 scripts/copy_handles_to_low_resource.py [handles_path] [dest_dir] [--dry-run]

"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable, List, Optional, Tuple


EXTS = [".flac", ".wav", ".mp3"]


def iso_ts() -> str:
    return datetime.now().isoformat()


def find_repo_root(script_path: Path, handles_arg: Optional[Path]) -> Path:
    # Look for ancestor named mls_*
    cur = script_path.resolve()
    for p in [cur] + list(cur.parents):
        if p.name.startswith("mls_"):
            return p

    # If handles arg provided, try to infer by walking up to 'train' parent
    if handles_arg:
        try:
            hp = handles_arg.resolve()
        except Exception:
            hp = Path(handles_arg)
        for p in [hp] + list(hp.parents):
            if p.name == "train":
                return p.parent

    # Fallback: script directory's parent (assume script is in <repo>/scripts)
    return script_path.resolve().parents[1] if len(script_path.resolve().parents) >= 2 else script_path.resolve().parent


def derive_language(repo_root: Path, dest_arg: Optional[Path]) -> str:
    b = repo_root.name
    if b.startswith("mls_"):
        return b[len("mls_"):]
    if dest_arg:
        t = dest_arg.name
        if t.endswith("_low_resource"):
            return t[: -len("_low_resource")]
        return t
    return b


def parse_handles(lines: Iterable[str]) -> Iterable[str]:
    for raw in lines:
        line = raw.strip("\n\r \t")
        if not line:
            continue
        if line.startswith("#"):
            continue
        yield line


def find_audio(repo_root: Path, handle: str, exts: List[str]) -> Optional[Path]:
    # expect handle like speaker_recording_segment
    parts = handle.split("_", 2)
    if len(parts) < 2:
        return None
    p1, p2 = parts[0], parts[1]
    src_dir = repo_root / "train" / "audio" / p1 / p2
    for e in exts:
        candidate = src_dir / (handle + e)
        if candidate.exists():
            return candidate
    return None


def append_log(logfile: Path, status: str, src: str, dest: str) -> None:
    header = "timestamp\tstatus\tsrc\tdest\n"
    if not logfile.exists():
        logfile.parent.mkdir(parents=True, exist_ok=True)
        logfile.write_text(header)
    with logfile.open("a", encoding="utf-8") as f:
        f.write(f"{iso_ts()}\t{status}\t{src}\t{dest}\n")


def copy_no_clobber(src: Path, dst: Path, dry_run: bool) -> Tuple[int, str]:
    """Copy src -> dst without overwriting.

    Returns (rc, message)
      rc=0: copied
      rc=1: skipped because dst exists
      rc=2: failed to copy
    """
    if dry_run:
        return 0, "dry-run"
    if dst.exists():
        return 1, "exists"
    try:
        shutil.copy2(src, dst)
        return 0, "copied"
    except Exception as e:  # noqa: BLE001 - we want to catch and log
        return 2, str(e)


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Copy handles -> low resource audio + transcripts")
    p.add_argument("handles", nargs="?", help="Path to handles file (default: train/limited_supervision/9hr/handles.txt)")
    p.add_argument("dest", nargs="?", help="Destination directory (default: <repo>/<language>_low_resource)")
    p.add_argument("-n", "--dry-run", action="store_true", help="Dry run: don't actually copy files")
    p.add_argument("--ext", action="append", help="Audio extension to probe (may be given multiple times). Example: --ext .flac --ext .wav")
    p.add_argument("--limit", type=int, default=0, help="Optional: limit number of handles processed (0 = no limit)")
    args = p.parse_args(argv)

    script_path = Path(__file__).resolve()
    handles_arg = Path(args.handles) if args.handles else None
    repo_root = find_repo_root(script_path.parent, handles_arg)
    language = derive_language(repo_root, Path(args.dest) if args.dest else None)

    handles_path = Path(args.handles) if args.handles else repo_root / "train" / "limited_supervision" / "9hr" / "handles.txt"
    dest_dir = Path(args.dest) if args.dest else repo_root / f"{language}_low_resource"
    exts = args.ext if args.ext else EXTS

    if not handles_path.exists():
        print(f"ERROR: handles file not found: {handles_path}", file=sys.stderr)
        return 2

    print(f"Repo root: {repo_root}")
    print(f"Language: {language}")
    print(f"Handles file: {handles_path}")
    print(f"Destination: {dest_dir}")
    if args.dry_run:
        print("Dry run: no files will be copied")

    log_file = dest_dir / "copy_log.tsv"
    append_log(log_file, "INFO", "-", "Started copy run")

    count_total = 0
    count_copied = 0
    count_missing = 0

    with handles_path.open("r", encoding="utf-8", errors="ignore") as hf:
        for line in parse_handles(hf):
            if args.limit and count_total >= args.limit:
                break
            count_total += 1
            handle = line
            audio_src = find_audio(repo_root, handle, exts)
            if audio_src is None:
                print(f"Missing source for {handle}", file=sys.stderr)
                count_missing += 1
                append_log(log_file, "MISSING", handle, "-")
                continue

            # preserve structure under dest_dir/audio/<speaker>/<recording>/
            parts = handle.split("_", 2)
            p1, p2 = parts[0], parts[1]
            dest_path = dest_dir / "audio" / p1 / p2 / audio_src.name
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if args.dry_run:
                print(f"Would copy: {audio_src} -> {dest_path}")
                append_log(log_file, "DRY-RUN", str(audio_src), str(dest_path))
                count_copied += 1
            else:
                rc, msg = copy_no_clobber(audio_src, dest_path, dry_run=False)
                if rc == 0:
                    append_log(log_file, "COPIED", str(audio_src), str(dest_path))
                    count_copied += 1
                elif rc == 1:
                    append_log(log_file, "SKIPPED-EXISTS", str(audio_src), str(dest_path))
                else:
                    print(f"Failed to copy {audio_src}: {msg}", file=sys.stderr)
                    append_log(log_file, "FAILED", str(audio_src), str(dest_path))

            # transcripts: prefer .lab then .txt; write dest as .lab
            base_noext = audio_src.with_suffix("")
            transcript_src = None
            lab_src = base_noext.with_suffix(".lab")
            txt_src = base_noext.with_suffix(".txt")
            if lab_src.exists():
                transcript_src = lab_src
            elif txt_src.exists():
                transcript_src = txt_src

            transcript_dest = dest_path.parent / (audio_src.stem + ".lab")
            if transcript_src:
                if args.dry_run:
                    append_log(log_file, "TRANSCRIPT-DRY", str(transcript_src), str(transcript_dest))
                else:
                    if transcript_dest.exists():
                        append_log(log_file, "TRANSCRIPT-SKIPPED-EXISTS", str(transcript_src), str(transcript_dest))
                    else:
                        try:
                            shutil.copy2(transcript_src, transcript_dest)
                            append_log(log_file, "TRANSCRIPT-COPIED", str(transcript_src), str(transcript_dest))
                        except Exception as e:
                            append_log(log_file, "TRANSCRIPT-FAILED", str(transcript_src), str(transcript_dest))
            else:
                append_log(log_file, "TRANSCRIPT-MISSING", handle, "-")

    append_log(log_file, "INFO", "-", f"Finished copy run: total={count_total},copied={count_copied},missing={count_missing}")
    print(f"Done. Total lines: {count_total}, copied: {count_copied}, missing: {count_missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

