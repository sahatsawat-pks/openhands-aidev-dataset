Remove rows matching a repository name or repo_id
===============================================

Purpose
-------
Small utility to remove rows from CSV files where the repository equals a given name (e.g. `drivly/ai`) or `repo_id` equals a given id (e.g. `950132973.0`).

Usage
-----
Run the script with one or more input files or glob patterns. Example:

```bash
python Script/remove_repo_rows.py --input "AIDev-Time/*.csv" \
  --repo-name drivly/ai --repo-id 950132973.0 --output-dir filtered/
```

This writes filtered files into `filtered/` with the `.filtered.csv` suffix. To overwrite input files, use `--inplace` (use carefully).

Options
-------
- `--repo-column`: column name for repo full name (default: `repo`)
- `--repo-id-column`: column name for repo id (default: `repo_id`)
- `--suffix`: output filename suffix (default: `.filtered.csv`)

Examples
--------
Filter a single file and write next to it:

```bash
python Script/remove_repo_rows.py -i AIDev-Time/AIDev-Time-Top4-Timestamps.csv \
  --repo-name drivly/ai --repo-id 950132973.0
```

Filter multiple files using glob:

```bash
python Script/remove_repo_rows.py -i "AIDev-Time/*.csv" -o filtered/ --repo-id 950132973.0
```
