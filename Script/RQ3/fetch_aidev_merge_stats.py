"""
Fetch global merge statistics (merged PR count and total PR count) for AIDev repositories
using GitHub Search API. This calculates the *project-level* merge rate.
"""

import pandas as pd
import requests
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[2]

# Allow overriding with environment variables (fall back to repo-relative defaults)
INPUT_FILE = Path(os.getenv('AIDEV_FULL_CSV')) if os.getenv('AIDEV_FULL_CSV') \
             else REPO_ROOT / "AIDev-Full" / "AIDev-PRs-Full.csv"

OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR')) if os.getenv('OUTPUT_DIR') \
             else REPO_ROOT / "analysis_outputs" / "project_comparison"

OUTPUT_FILE = OUTPUT_DIR / "AIDev-PRs-Full-with-merge-stats.csv"

# GitHub API configuration
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
HEADERS = {
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28'
}
if GITHUB_TOKEN:
    HEADERS['Authorization'] = f'Bearer {GITHUB_TOKEN}'

def get_repo_full_name(api_url):
    """Extract owner/repo from API URL"""
    if 'api.github.com/repos/' in api_url:
        return api_url.split('api.github.com/repos/')[1]
    return None

def fetch_repo_stats(repo_full_name, retries=3):
    """
    Fetch merged and total PR counts for a repository using Search API.
    Returns: (merged_count, total_count)
    """
    search_url = "https://api.github.com/search/issues"
    
    # query 1: merged
    q_merged = f"repo:{repo_full_name} is:pr is:merged"
    merged = _fetch_count(search_url, q_merged, retries)
    
    # query 2: total (is:pr)
    q_total = f"repo:{repo_full_name} is:pr"
    total = _fetch_count(search_url, q_total, retries)
    
    return merged, total

def fetch_time_window_stats(repo_full_name, start_date="2024-03-12", end_date="2025-10-16", retries=3):
    """
    Fetch merged and total PR counts for a repository within a specific time window.
    Query: created:START..END
    Returns: (merged_count, total_count)
    """
    search_url = "https://api.github.com/search/issues"
    date_range = f"{start_date}..{end_date}"
    
    # query 1: merged in window
    q_merged = f"repo:{repo_full_name} is:pr is:merged created:{date_range}"
    merged = _fetch_count(search_url, q_merged, retries)
    
    # query 2: total in window
    q_total = f"repo:{repo_full_name} is:pr created:{date_range}"
    total = _fetch_count(search_url, q_total, retries)
    
    return merged, total

def _fetch_count(url, query, retries=3):
    params = {'q': query, 'per_page': 1}
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            if response.status_code == 200:
                return response.json().get('total_count', 0)
            elif response.status_code == 403:
                # Rate limit handling
                reset = response.headers.get('X-RateLimit-Reset')
                wait = max(int(reset) - time.time(), 0) + 1 if reset else 60
                print(f"  ⚠️ Rate limit. Waiting {wait:.0f}s...")
                time.sleep(wait)
            elif response.status_code == 422:
                print(f"  ❌ Validation failed (422)")
                return None
            else:
                print(f"  ❌ Error {response.status_code}")
                return None
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            time.sleep(2)
            
    return None

def main():
    print("=" * 80)
    print("Fetching Global & Time-Windowed Merge Stats for AIDev Repositories")
    print("=" * 80)
    
    if not GITHUB_TOKEN:
        print("\n⚠️  WARNING: No GitHub token found. Search API rate limit is very low (10/min)!")
        
    # Load data
    print(f"\nLoading data from: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Target Repositories (Top 5 from Full and Top 5 from TimeFiltered)
    target_ids = [985853139, 1021905497, 922805069, 975733848, 994985630, # AIDev-Full
                  922805069, 839640906, 950132973, 979695673, 982394988]  # AIDev-Time
    target_ids = [str(x) for x in list(set(target_ids))]
    
    # Ensure column is string for matching
    df['repo_id_group'] = df['repo_id_group'].astype(str)
    
    print(f"Filtering for {len(target_ids)} specific repositories...")
    df = df[df['repo_id_group'].isin(target_ids)].copy()
    print(f"Found {len(df)} matching repositories")
    
    # Columns for Global Stats
    if 'global_merged_pr_count' not in df.columns:
        df['global_merged_pr_count'] = None
    if 'global_total_pr_count' not in df.columns:
        df['global_total_pr_count'] = None
        
    # Columns for Time-Window Stats (March 2024 - Oct 2025)
    if 'window_merged_pr_count' not in df.columns:
        df['window_merged_pr_count'] = None
    if 'window_total_pr_count' not in df.columns:
        df['window_total_pr_count'] = None
    
    print("\n🔄 Fetching stats...")
    
    processed = 0
    for idx, row in df.iterrows():
        repo_url = row.get('repo_url_group', '')
        repo_full_name = get_repo_full_name(repo_url)
        
        print(f"\n[{processed+1}/{len(df)}] Processing: {repo_full_name}")
        
        if not repo_full_name:
            processed += 1
            continue
            
        # 1. Global Stats
        merged, total = fetch_repo_stats(repo_full_name)
        if merged is not None:
            df.at[idx, 'global_merged_pr_count'] = merged
            df.at[idx, 'global_total_pr_count'] = total
            rate = (merged/total*100) if total and total > 0 else 0
            print(f"  global: {merged}/{total} ({rate:.1f}%)")
        else:
            print(f"  ❌ Global Failed")
            
        # 2. Time Window Stats
        w_merged, w_total = fetch_time_window_stats(repo_full_name)
        if w_merged is not None:
            df.at[idx, 'window_merged_pr_count'] = w_merged
            df.at[idx, 'window_total_pr_count'] = w_total
            w_rate = (w_merged/w_total*100) if w_total and w_total > 0 else 0
            print(f"  window: {w_merged}/{w_total} ({w_rate:.1f}%)")
        else:
            print(f"  ❌ Window Failed")

        processed += 1
        time.sleep(2)
            
    print(f"\n✓ Fetch complete!")
    
    # Calculate rates
    for prefix in ['global', 'window']:
        df[f'{prefix}_merged_pr_count'] = pd.to_numeric(df[f'{prefix}_merged_pr_count'])
        df[f'{prefix}_total_pr_count'] = pd.to_numeric(df[f'{prefix}_total_pr_count'])
        
        df[f'{prefix}_merge_rate'] = df.apply(
            lambda row: (row[f'{prefix}_merged_pr_count'] / row[f'{prefix}_total_pr_count'] * 100)
            if (pd.notna(row[f'{prefix}_merged_pr_count']) and row[f'{prefix}_total_pr_count'] > 0)
            else None,
            axis=1
        )
    
    # Save
    print(f"Saving results to: {OUTPUT_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Print table for user
    print("\n" + "=" * 80)
    print("RESULTS TABLE (Sorted by Repo ID)")
    print("=" * 80)
    cols = ['repo_id_group', 'repo_url_group', 'global_merge_rate', 'window_merge_rate', 'window_total_pr_count']
    print(df[cols].to_string(index=False))

if __name__ == "__main__":
    main()
