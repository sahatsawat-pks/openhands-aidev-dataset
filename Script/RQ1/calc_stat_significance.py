import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def cliffs_delta(x, y):
    """
    Calculates Cliff's Delta statistic.
    d = ( #(x>y) - #(x<y) ) / (n*m)
    """
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return np.nan
        
    # Optimization for large arrays using numpy broadcasting is too memory heavy for >20k items (400M elements).
    # Use sorting approach or simple loop with numba (not available here).
    # Fallback to a slightly optimized approach.
    
    # Sort both arrays
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # This can still be slow. 
    # Let's use a faster approximation or Cohen's d if N is very large? 
    # Or strict bucket counting if values are integers (commit counts ARE integers!).
    
    # Bucket sort approach for integers (Commit counts max out but usually low)
    # Combine and find range
    all_vals = np.concatenate([x, y])
    min_val, max_val = int(all_vals.min()), int(all_vals.max())
    
    # If range is huge, this is bad. If range is small (e.g. 0-1000), this is instant.
    # Check range. 
    if max_val - min_val > 10000:
        # Fallback to Cohen's d if range is crazy
        return cohens_d(x, y)
        
    # Count frequencies
    x_counts = np.bincount(x.astype(int) - min_val, minlength=max_val-min_val+1)
    y_counts = np.bincount(y.astype(int) - min_val, minlength=max_val-min_val+1)
    
    # Accumulated counts
    total_greater = 0
    total_smaller = 0
    
    # Iterate through unique values in order
    # For each value v in x:
    #   smaller in y = sum(y_counts < v)
    #   greater in y = sum(y_counts > v)
    
    # Precompute cumsum for y
    y_cumsum = np.cumsum(y_counts)
    
    n_y = len(y)
    
    for v_idx, count_x in enumerate(x_counts):
        if count_x == 0: continue
        
        # y values smaller than current v_idx
        smaller = y_cumsum[v_idx-1] if v_idx > 0 else 0
        
        # y values greater than current v_idx
        # total y - (y <= v_idx) -> total y - y_cumsum[v_idx]
        greater = n_y - y_cumsum[v_idx]
        
        # x > y contribution: x=v, y < v.  (count_x * smaller)
        total_greater += count_x * smaller
        
        # x < y contribution: x=v, y > v. (count_x * greater)
        total_smaller += count_x * greater
    
    d = (total_greater - total_smaller) / (n * m)
    return d

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dOf = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dOf)

def interpret_cliffs(d):
    ad = abs(d)
    if ad < 0.147: return "Negligible"
    if ad < 0.33: return "Small"
    if ad < 0.474: return "Medium"
    return "Large"

def interpret_p(p):
    return "Significant" if p < 0.05 else "Not Significant"

def get_repo_name(url):
    try:
        return url.split('/')[-1]
    except:
        return "Unknown"

# Load Data
print("Loading data...")
df_oh = pd.read_csv('analysis_outputs/openhands_prs_2024-03-13_to_2025-06-11_with_commits.csv')
df_oh['repo'] = 'OpenHands'
oh_commits = df_oh['commit_count'].dropna().values

df_full = pd.read_csv('analysis_outputs/aidev_full_top5_prs_with_commit_size.csv')
df_full['repo'] = df_full['repo_url'].apply(get_repo_name)

df_time = pd.read_csv('analysis_outputs/aidev_time_top5_prs_with_commit_size.csv')
df_time['repo'] = df_time['repo_url'].apply(get_repo_name)

comparisons = []

# Helper Comparison
def run_compare(name, other_commits):
    other_commits = np.array(other_commits)
    other_commits = other_commits[~np.isnan(other_commits)]
    
    if len(other_commits) == 0:
        return
        
    stat, p = mannwhitneyu(oh_commits, other_commits, alternative='two-sided')
    d = cliffs_delta(oh_commits, other_commits)
    
    comparisons.append({
        'Comparison': f"OpenHands vs {name}",
        'OH Mean': np.mean(oh_commits),
        'Other Mean': np.mean(other_commits),
        'p-value': p,
        "Cliff's d": d,
        'Effect Size': interpret_cliffs(d)
    })

# 1. Aggregate
print("Calculating Aggregate Stats...")
run_compare('All AIDev Full', df_full['commit_count'])
run_compare('All AIDev Time', df_time['commit_count'])

# 2. Per Repo (Full)
print("Calculating Per-Repo (Full) Stats...")
for repo in df_full['repo'].unique():
    subset = df_full[df_full['repo'] == repo]['commit_count']
    run_compare(f"Full/{repo}", subset)

# 3. Per Repo (Time)
print("Calculating Per-Repo (Time) Stats...")
for repo in df_time['repo'].unique():
    subset = df_time[df_time['repo'] == repo]['commit_count']
    run_compare(f"Time/{repo}", subset)

# Display Table
df_res = pd.DataFrame(comparisons)
print("\n--- Statistical Analysis: Commits per PR ---")
# Format for readable output
# Use scientific notation for p-values
pd.set_option('display.float_format', lambda x: '%.3e' % x if abs(x) < 0.01 else '%.4f' % x)
print(df_res.to_string())

# Save to CSV
df_res.to_csv('analysis_outputs/stats_commits_comparison.csv', index=False)
