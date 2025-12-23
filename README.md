OpenHands AIDev Dataset
=======================

Short description
-----------------
This repository contains datasets and analysis scripts used for studying PR and commit patterns across multiple projects (AIDev and OpenHands datasets).

Repository layout
-----------------
- `AIDev-Full/` — full AIDev dataset CSVs.
- `AIDev-Time/` — time-filtered AIDev dataset CSVs.
- `OpenHands/` — OpenHands PR datasets.
- `Script/` — analysis scripts organized by research question (RQ1, RQ2, RQ3).

Key scripts
-----------
- `Script/RQ1/rq1_full_analysis.py` — RQ1 analyses and helpers (requires pandas and other analysis packages).
- `Script/RQ2/rq2_temporal_analysis.py` — temporal analysis for RQ2.
- `Script/RQ1/feature_extraction.py`, `Script/RQ1/visualization_dashboard.py`, etc. — additional helpers.

Data files
----------
Source CSVs are inside `AIDev-Full/`, `AIDev-Time/`, and `OpenHands/`. Scripts are written to use repo-relative paths (no absolute `/Users/...` paths are required).

Dependencies
------------
- Python 3.8+ recommended
- Minimal Python packages (install via pip):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you don't have `requirements.txt`, install the essentials:

```bash
pip install pandas requests python-dotenv
```

Environment
-----------
This repo includes a `.env.template` at the project root. Copy it to `.env` and fill in sensitive or environment-specific values (for example your `GITHUB_TOKEN` and any custom input/output paths). Scripts in `Script/` will read values from your `.env` if present.

```bash
cp .env.template .env
# edit .env to add your token and paths
```

Running the scripts
-------------------
Example: run the GitHub fetch script (set your token to avoid strict rate limits):

```bash
export GITHUB_TOKEN="ghp_..."   # set your token securely
python Script/RQ3/fetch_aidev_merge_stats.py
```

Other scripts can be run similarly, e.g.:

```bash
python Script/RQ1/rq1_full_analysis.py
python Script/RQ2/rq2_temporal_analysis.py
```

Notes
-----
- `Script/RQ3/fetch_aidev_merge_stats.py` now uses repo-relative paths and will create `analysis_outputs/project_comparison/` before writing output.
- Some CSV files may contain incidental `/Users/...` strings as data/content; those are not used as script configuration.

Contributing / Next steps
-------------------------
If you want, I can:
- Convert other scripts to use repo-relative paths.
- Add a `requirements.txt` capturing exact packages used by all scripts.
- Add small run examples for each RQ script.
