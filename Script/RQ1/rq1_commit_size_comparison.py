"""
RQ1: Commit Size Comparison (Additions and Deletions)
Compare OpenHands with top 5 repos in AI-Dev Full and AI-Dev Time datasets
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
openhands = openhands.dropna(subset=['additions', 'deletions'])
aidev_full = aidev_full.dropna(subset=['additions', 'deletions'])
aidev_time = aidev_time.dropna(subset=['additions', 'deletions'])

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

# Calculate OpenHands baseline statistics
oh_stats = {
    'additions_mean': openhands['additions'].mean(),
    'deletions_mean': openhands['deletions'].mean(),
    'additions_median': openhands['additions'].median(),
    'deletions_median': openhands['deletions'].median(),
    'additions_data': openhands['additions'],
    'deletions_data': openhands['deletions']
}

print("\n" + "="*80)
print("OPENHANDS BASELINE")
print("="*80)
print(f"Additions/PR:  mean={oh_stats['additions_mean']:.2f}, median={oh_stats['additions_median']:.2f}")
print(f"Deletions/PR:  mean={oh_stats['deletions_mean']:.2f}, median={oh_stats['deletions_median']:.2f}")

def cohen_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def interpret_effect(d):
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"

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

# Analyze AI-Dev Full repositories
print("\n" + "="*80)
print("AI-DEV FULL REPOSITORIES")
print("="*80)

full_results = []
for repo_id, repo_name in sorted(repo_mapping.items()):
    repo_data = aidev_full[aidev_full['repo_id'] == repo_id]
    if len(repo_data) == 0:
        continue
    
    # Additions
    additions = repo_data['additions']
    additions_mean = additions.mean()
    _, additions_p = stats.mannwhitneyu(oh_stats['additions_data'], additions, alternative='two-sided')
    additions_d = cohen_d(oh_stats['additions_data'], additions)
    
    # Deletions
    deletions = repo_data['deletions']
    deletions_mean = deletions.mean()
    _, deletions_p = stats.mannwhitneyu(oh_stats['deletions_data'], deletions, alternative='two-sided')
    deletions_d = cohen_d(oh_stats['deletions_data'], deletions)
    
    full_results.append({
        'dataset': 'AIDev-All',
        'repo_name': repo_name,
        'n_prs': len(repo_data),
        'additions_mean': additions_mean,
        'additions_d': additions_d,
        'additions_sig': format_pvalue(additions_p),
        'additions_effect': interpret_effect(additions_d),
        'deletions_mean': deletions_mean,
        'deletions_d': deletions_d,
        'deletions_sig': format_pvalue(deletions_p),
        'deletions_effect': interpret_effect(deletions_d)
    })
    
    print(f"\n{repo_name} (n={len(repo_data)}):")
    print(f"  Additions/PR: {additions_mean:.2f} (d={additions_d:.4f}, {interpret_effect(additions_d)}){format_pvalue(additions_p)}")
    print(f"  Deletions/PR: {deletions_mean:.2f} (d={deletions_d:.4f}, {interpret_effect(deletions_d)}){format_pvalue(deletions_p)}")

# Analyze AI-Dev Time repositories
print("\n" + "="*80)
print("AI-DEV TIME REPOSITORIES")
print("="*80)

time_results = []
for repo_id, repo_name in sorted(repo_mapping.items()):
    repo_data = aidev_time[aidev_time['repo_id'] == repo_id]
    if len(repo_data) == 0:
        continue
    
    # Additions
    additions = repo_data['additions']
    additions_mean = additions.mean()
    _, additions_p = stats.mannwhitneyu(oh_stats['additions_data'], additions, alternative='two-sided')
    additions_d = cohen_d(oh_stats['additions_data'], additions)
    
    # Deletions
    deletions = repo_data['deletions']
    deletions_mean = deletions.mean()
    _, deletions_p = stats.mannwhitneyu(oh_stats['deletions_data'], deletions, alternative='two-sided')
    deletions_d = cohen_d(oh_stats['deletions_data'], deletions)
    
    time_results.append({
        'dataset': 'AIDev-Time',
        'repo_name': repo_name,
        'n_prs': len(repo_data),
        'additions_mean': additions_mean,
        'additions_d': additions_d,
        'additions_sig': format_pvalue(additions_p),
        'additions_effect': interpret_effect(additions_d),
        'deletions_mean': deletions_mean,
        'deletions_d': deletions_d,
        'deletions_sig': format_pvalue(deletions_p),
        'deletions_effect': interpret_effect(deletions_d)
    })
    
    print(f"\n{repo_name} (n={len(repo_data)}):")
    print(f"  Additions/PR: {additions_mean:.2f} (d={additions_d:.4f}, {interpret_effect(additions_d)}){format_pvalue(additions_p)}")
    print(f"  Deletions/PR: {deletions_mean:.2f} (d={deletions_d:.4f}, {interpret_effect(deletions_d)}){format_pvalue(deletions_p)}")

# Generate LaTeX table for Additions
print("\n" + "="*80)
print("GENERATING LATEX TABLES")
print("="*80)

latex_additions = r"""\begin{table}[h]
\centering
\caption{Comparison of Additions per PR}
\label{tab:additions_per_pr}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcc}
\toprule
\textbf{Dataset} & \textbf{Repo Name} & \textbf{Additions/PR} & \textbf{Stat. Diff} ($d$) \\
\midrule
OpenHands & OpenHands & """ + f"{oh_stats['additions_mean']:.2f}" + r""" & - \\
\midrule
"""

# Add AI-Dev Full results
for i, r in enumerate(full_results):
    if i == 0:
        latex_additions += f"AIDev-All & {r['repo_name']} & {r['additions_mean']:.2f} & ({r['additions_sig']}) \\\\\n"
    else:
        latex_additions += f" & {r['repo_name']} & {r['additions_mean']:.2f} & ({r['additions_sig']}) \\\\\n"

latex_additions += r"""\midrule
"""

# Add AI-Dev Time results
for i, r in enumerate(time_results):
    if i == 0:
        latex_additions += f"AIDev-Time & {r['repo_name']} & {r['additions_mean']:.2f} & ({r['additions_sig']}) \\\\\n"
    else:
        latex_additions += f" & {r['repo_name']} & {r['additions_mean']:.2f} & ({r['additions_sig']}) \\\\\n"

latex_additions += r"""
\bottomrule
\end{tabular}%
}
\begin{flushleft}
\small
Stat. Diff: Mann-Whitney U test relative to OpenHands (* $< 0.05$, ** $< 0.01$, *** $< 0.001$)
\end{flushleft}
\end{table}
"""

# Generate LaTeX table for Deletions
latex_deletions = r"""\begin{table}[h]
\centering
\caption{Comparison of Deletions per PR}
\label{tab:deletions_per_pr}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcc}
\toprule
\textbf{Dataset} & \textbf{Repo Name} & \textbf{Deletions/PR} & \textbf{Stat. Diff} ($d$) \\
\midrule
OpenHands & OpenHands & """ + f"{oh_stats['deletions_mean']:.2f}" + r""" & - \\
\midrule
"""

# Add AI-Dev Full results
for i, r in enumerate(full_results):
    if i == 0:
        latex_deletions += f"AIDev-All & {r['repo_name']} & {r['deletions_mean']:.2f} & ({r['deletions_sig']}) \\\\\n"
    else:
        latex_deletions += f" & {r['repo_name']} & {r['deletions_mean']:.2f} & ({r['deletions_sig']}) \\\\\n"

latex_deletions += r"""\midrule
"""

# Add AI-Dev Time results
for i, r in enumerate(time_results):
    if i == 0:
        latex_deletions += f"AIDev-Time & {r['repo_name']} & {r['deletions_mean']:.2f} & ({r['deletions_sig']}) \\\\\n"
    else:
        latex_deletions += f" & {r['repo_name']} & {r['deletions_mean']:.2f} & ({r['deletions_sig']}) \\\\\n"

latex_deletions += r"""
\bottomrule
\end{tabular}%
}
\begin{flushleft}
\small
Stat. Diff: Mann-Whitney U test relative to OpenHands (* $< 0.05$, ** $< 0.01$, *** $< 0.001$)
\end{flushleft}
\end{table}
"""

# Save LaTeX tables
with open('analysis_outputs/rq1_additions_repo_comparison.tex', 'w') as f:
    f.write(latex_additions)
print("✓ Saved: analysis_outputs/rq1_additions_repo_comparison.tex")

with open('analysis_outputs/rq1_deletions_repo_comparison.tex', 'w') as f:
    f.write(latex_deletions)
print("✓ Saved: analysis_outputs/rq1_deletions_repo_comparison.tex")

# Save results to CSV
full_df = pd.DataFrame(full_results)
time_df = pd.DataFrame(time_results)

full_df.to_csv('analysis_outputs/rq1_commit_size_aidev_full.csv', index=False)
time_df.to_csv('analysis_outputs/rq1_commit_size_aidev_time.csv', index=False)

print("✓ Saved: analysis_outputs/rq1_commit_size_aidev_full.csv")
print("✓ Saved: analysis_outputs/rq1_commit_size_aidev_time.csv")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
