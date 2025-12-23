"""
RQ1: Repository-level comparison with OpenHands
Create tables similar to paper format showing individual repos
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load datasets
print("Loading datasets...")
openhands = pd.read_csv('analysis_outputs/openhands_prs_with_commit_size.csv')
aidev_full = pd.read_csv('analysis_outputs/aidev_full_top5_prs_with_commit_size.csv')
aidev_time = pd.read_csv('analysis_outputs/aidev_time_top5_prs_with_commit_size.csv')

# Remove nulls
openhands = openhands.dropna(subset=['commit_count', 'additions', 'deletions', 'changed_files'])
aidev_full = aidev_full.dropna(subset=['commit_count', 'additions', 'deletions', 'changed_files'])
aidev_time = aidev_time.dropna(subset=['commit_count', 'additions', 'deletions', 'changed_files'])

# Repository mapping
repo_mapping = {
    985853139.0: 'mochi',
    1021905497.0: 'codeforces',
    922805069.0: 'AGI-Alpha-Agent-v0',
    975733848.0: 'MVP-website',
    994985630.0: 'TonPlaygramWebApp',
    979695673.0: 'LemmingsJS-MIDI',
    839640906.0: 'glimpser',
    982394988.0: 'alfe-ai'
}

aidev_full['repo_name'] = aidev_full['repo_id'].map(repo_mapping)
aidev_time['repo_name'] = aidev_time['repo_id'].map(repo_mapping)

def cohen_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def format_pvalue(p):
    """Format p-value with significance stars"""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

# Calculate OpenHands baseline statistics
oh_commits = openhands['commit_count'].mean()
oh_additions = openhands['additions'].mean()
oh_deletions = openhands['deletions'].mean()
oh_files = openhands['changed_files'].mean()

print("\n" + "="*80)
print("OPENHANDS BASELINE")
print("="*80)
print(f"Commits/PR: {oh_commits:.2f}")
print(f"Additions/PR: {oh_additions:.2f}")
print(f"Deletions/PR: {oh_deletions:.2f}")
print(f"Files/PR: {oh_files:.2f}")

# Analyze AI-Dev Full repositories
print("\n" + "="*80)
print("AI-DEV FULL REPOSITORIES")
print("="*80)

full_results = []
for repo_id, repo_name in sorted(repo_mapping.items()):
    repo_data = aidev_full[aidev_full['repo_id'] == repo_id]
    if len(repo_data) == 0:
        continue
    
    # Commits/PR
    commits_mean = repo_data['commit_count'].mean()
    _, commits_p = stats.mannwhitneyu(repo_data['commit_count'], openhands['commit_count'])
    commits_d = cohen_d(repo_data['commit_count'], openhands['commit_count'])
    
    # Additions/PR
    additions_mean = repo_data['additions'].mean()
    _, additions_p = stats.mannwhitneyu(repo_data['additions'], openhands['additions'])
    additions_d = cohen_d(repo_data['additions'], openhands['additions'])
    
    # Deletions/PR
    deletions_mean = repo_data['deletions'].mean()
    _, deletions_p = stats.mannwhitneyu(repo_data['deletions'], openhands['deletions'])
    deletions_d = cohen_d(repo_data['deletions'], openhands['deletions'])
    
    # Files/PR
    files_mean = repo_data['changed_files'].mean()
    _, files_p = stats.mannwhitneyu(repo_data['changed_files'], openhands['changed_files'])
    files_d = cohen_d(repo_data['changed_files'], openhands['changed_files'])
    
    full_results.append({
        'dataset': 'AIDev-All',
        'repo_name': repo_name,
        'n_prs': len(repo_data),
        'commits_mean': commits_mean,
        'commits_d': commits_d,
        'commits_sig': format_pvalue(commits_p),
        'additions_mean': additions_mean,
        'additions_d': additions_d,
        'additions_sig': format_pvalue(additions_p),
        'deletions_mean': deletions_mean,
        'deletions_d': deletions_d,
        'deletions_sig': format_pvalue(deletions_p),
        'files_mean': files_mean,
        'files_d': files_d,
        'files_sig': format_pvalue(files_p)
    })
    
    print(f"\n{repo_name} (n={len(repo_data)}):")
    print(f"  Commits/PR: {commits_mean:.2f} (d={commits_d:.4f}{format_pvalue(commits_p)})")
    print(f"  Additions/PR: {additions_mean:.2f} (d={additions_d:.4f}{format_pvalue(additions_p)})")
    print(f"  Deletions/PR: {deletions_mean:.2f} (d={deletions_d:.4f}{format_pvalue(deletions_p)})")
    print(f"  Files/PR: {files_mean:.2f} (d={files_d:.4f}{format_pvalue(files_p)})")

# Analyze AI-Dev Time repositories
print("\n" + "="*80)
print("AI-DEV TIME REPOSITORIES")
print("="*80)

time_results = []
for repo_id, repo_name in sorted(repo_mapping.items()):
    repo_data = aidev_time[aidev_time['repo_id'] == repo_id]
    if len(repo_data) == 0:
        continue
    
    # Commits/PR
    commits_mean = repo_data['commit_count'].mean()
    _, commits_p = stats.mannwhitneyu(repo_data['commit_count'], openhands['commit_count'])
    commits_d = cohen_d(repo_data['commit_count'], openhands['commit_count'])
    
    # Additions/PR
    additions_mean = repo_data['additions'].mean()
    _, additions_p = stats.mannwhitneyu(repo_data['additions'], openhands['additions'])
    additions_d = cohen_d(repo_data['additions'], openhands['additions'])
    
    # Deletions/PR
    deletions_mean = repo_data['deletions'].mean()
    _, deletions_p = stats.mannwhitneyu(repo_data['deletions'], openhands['deletions'])
    deletions_d = cohen_d(repo_data['deletions'], openhands['deletions'])
    
    # Files/PR
    files_mean = repo_data['changed_files'].mean()
    _, files_p = stats.mannwhitneyu(repo_data['changed_files'], openhands['changed_files'])
    files_d = cohen_d(repo_data['changed_files'], openhands['changed_files'])
    
    time_results.append({
        'dataset': 'AIDev-Time',
        'repo_name': repo_name,
        'n_prs': len(repo_data),
        'commits_mean': commits_mean,
        'commits_d': commits_d,
        'commits_sig': format_pvalue(commits_p),
        'additions_mean': additions_mean,
        'additions_d': additions_d,
        'additions_sig': format_pvalue(additions_p),
        'deletions_mean': deletions_mean,
        'deletions_d': deletions_d,
        'deletions_sig': format_pvalue(deletions_p),
        'files_mean': files_mean,
        'files_d': files_d,
        'files_sig': format_pvalue(files_p)
    })
    
    print(f"\n{repo_name} (n={len(repo_data)}):")
    print(f"  Commits/PR: {commits_mean:.2f} (d={commits_d:.4f}{format_pvalue(commits_p)})")
    print(f"  Additions/PR: {additions_mean:.2f} (d={additions_d:.4f}{format_pvalue(additions_p)})")
    print(f"  Deletions/PR: {deletions_mean:.2f} (d={deletions_d:.4f}{format_pvalue(deletions_p)})")
    print(f"  Files/PR: {files_mean:.2f} (d={files_d:.4f}{format_pvalue(files_p)})")

# Create combined markdown table
markdown = f"""
# RQ1: Repository-Level Comparison with OpenHands

## Table: Commits per PR

| Dataset | Repo Name | Commits/PR | Effect Size (d) | Significance |
|---------|-----------|------------|-----------------|--------------|
| OpenHands | OpenHands | {oh_commits:.2f} | - | - |
"""

for r in full_results:
    markdown += f"| AIDev-All | {r['repo_name']} | {r['commits_mean']:.2f} | {r['commits_d']:.4f} | {r['commits_sig']} |\n"

for r in time_results:
    markdown += f"| AIDev-Time | {r['repo_name']} | {r['commits_mean']:.2f} | {r['commits_d']:.4f} | {r['commits_sig']} |\n"

markdown += f"""
## Table: Files Changed per PR

| Dataset | Repo Name | Files/PR | Effect Size (d) | Significance |
|---------|-----------|----------|-----------------|--------------|
| OpenHands | OpenHands | {oh_files:.2f} | - | - |
"""

for r in full_results:
    markdown += f"| AIDev-All | {r['repo_name']} | {r['files_mean']:.2f} | {r['files_d']:.4f} | {r['files_sig']} |\n"

for r in time_results:
    markdown += f"| AIDev-Time | {r['repo_name']} | {r['files_mean']:.2f} | {r['files_d']:.4f} | {r['files_sig']} |\n"

markdown += """
**Notes:**
- \\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001 (Mann-Whitney U test)
- Effect sizes (Cohen's d) compared to OpenHands baseline
"""

with open('analysis_outputs/rq1_repo_comparison_summary.md', 'w') as f:
    f.write(markdown)
print("✓ Saved: analysis_outputs/rq1_repo_comparison_summary.md")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
