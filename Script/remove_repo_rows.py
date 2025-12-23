#!/usr/bin/env python3
"""Filter CSV files to remove rows matching a repository name or repo_id.

Example:
  python Script/remove_repo_rows.py --input AIDev-Time/AIDev-Time-Top4-Timestamps.csv \
      --repo-name drivly/ai --repo-id 950132973.0 --output-dir filtered/

This script writes a filtered file per input with suffix `.filtered.csv` by default.
"""
import argparse
import os
import glob
import pandas as pd
from typing import List


def parse_args():
    p = argparse.ArgumentParser(description="Remove rows matching a repo name or repo_id from CSV files")
    p.add_argument("--input", "-i", nargs="+", required=True,
                   help="Input file path(s) or glob pattern(s).")
    p.add_argument("--repo-name", help="Repository full name to remove (e.g. drivly/ai)")
    p.add_argument("--repo-id", help="Repository id to remove (e.g. 950132973.0)")
    p.add_argument("--repo-column", default="repo", help="Column name that holds repository full name (default: repo)")
    p.add_argument("--repo-id-column", default="repo_id", help="Column name that holds repo id (default: repo_id)")
    p.add_argument("--output-dir", "-o", default=None, help="Directory to write filtered files. If omitted, writes next to input files.")
    p.add_argument("--inplace", action="store_true", help="Overwrite input files with filtered output (use with caution)")
    p.add_argument("--suffix", default=".filtered.csv", help="Suffix appended to filtered filenames (default: .filtered.csv)")
    return p.parse_args()


def expand_inputs(patterns: List[str]) -> List[str]:
    files = []
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            files.extend(matches)
        elif os.path.isfile(p):
            files.append(p)
    # dedupe while preserving order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def repo_id_matches(cell, target: str) -> bool:
    if pd.isna(cell):
        return False
    try:
        # compare numerically if possible
        return float(cell) == float(target)
    except Exception:
        # fallback to string compare
        return str(cell).strip() == str(target).strip()


def filter_file(path: str, repo_name: str, repo_id: str, repo_col: str, repo_id_col: str, out_path: str):
    df = pd.read_csv(path, dtype=str, low_memory=False)

    mask = pd.Series(False, index=df.index)
    if repo_name:
        if repo_col not in df.columns:
            print(f"Warning: column '{repo_col}' not in {path}; skipping repo-name match")
        else:
            mask = mask | (df[repo_col] == repo_name)
    if repo_id:
        if repo_id_col not in df.columns:
            print(f"Warning: column '{repo_id_col}' not in {path}; skipping repo-id match")
        else:
            matches = df[repo_id_col].apply(lambda c: repo_id_matches(c, repo_id))
            mask = mask | matches

    filtered = df.loc[~mask]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    filtered.to_csv(out_path, index=False)
    kept = len(filtered)
    removed = int(mask.sum())
    print(f"Wrote {out_path} — kept {kept}, removed {removed}")


def main():
    args = parse_args()
    inputs = expand_inputs(args.input)
    if not inputs:
        print("No input files found for the given patterns.")
        return
    for inp in inputs:
        dirname, basename = os.path.split(inp)
        if args.inplace and (args.output_dir is None):
            out_path = inp
        else:
            out_dir = args.output_dir or dirname
            base, _ = os.path.splitext(basename)
            out_name = base + args.suffix
            out_path = os.path.join(out_dir, out_name)
        filter_file(inp, args.repo_name, args.repo_id, args.repo_column, args.repo_id_column, out_path)


if __name__ == "__main__":
    main()
