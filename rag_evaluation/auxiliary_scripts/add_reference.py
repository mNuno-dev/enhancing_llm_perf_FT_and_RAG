#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

def process_jsonl(src_file: Path, dst_file: Path, overwrite_reference: bool) -> None:
    dst_file.parent.mkdir(parents=True, exist_ok=True)

    with src_file.open("r", encoding="utf-8") as fin, dst_file.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                fout.write("\n")
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error in {src_file} at line {line_no}: {e}") from e

            if isinstance(obj, dict) and "answer" in obj:
                if overwrite_reference or "reference" not in obj:
                    obj["reference"] = obj["answer"]

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

def replicate_and_add_reference(src_root: Path, dst_root: Path, overwrite_reference: bool) -> None:
    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(f"Source folder does not exist or is not a directory: {src_root}")

    # Create base destination
    dst_root.mkdir(parents=True, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(src_root):
        rel_dir = Path(dirpath).relative_to(src_root)
        out_dir = dst_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for fname in filenames:
            src_path = Path(dirpath) / fname
            dst_path = out_dir / fname

            if fname.lower().endswith(".jsonl"):
                process_jsonl(src_path, dst_path, overwrite_reference)
            else:
                # Copy other files byte-for-byte if any exist
                dst_path.write_bytes(src_path.read_bytes())

def main():
    ap = argparse.ArgumentParser(
        description="Replicate a folder tree and add 'reference' = 'answer' to each JSONL entry."
    )
    ap.add_argument(
        "--src",
        required=True,
        help="Source folder"
    )
    ap.add_argument(
        "--dst",
        default=None,
        help="Destination folder (default: <src>_with_reference)"
    )
    ap.add_argument(
        "--overwrite-reference",
        action="store_true",
        help="If set, overwrite existing 'reference' keys. Otherwise only add when missing."
    )
    args = ap.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve() if args.dst else Path(str(src_root) + "_with_reference").resolve()

    if dst_root.exists() and any(dst_root.iterdir()):
        raise SystemExit(f"Destination folder already exists and is not empty: {dst_root}")

    replicate_and_add_reference(src_root, dst_root, args.overwrite_reference)
    print(f"Done. Wrote replicated folder to: {dst_root}")

if __name__ == "__main__":
    main()