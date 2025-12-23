"""
RQ3: Time to Merge Analysis

Examines pull request merge rates, time to merge, and collaboration patterns.
Includes comprehensive visualizations and statistical comparisons following
the methodology from paper.tex.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load PR data and calculate time to merge"""
    print("Loading datasets...")
    
    # Allow overriding input filenames via environment variables (.env)
    load_dotenv()
    openhands_csv = os.getenv('OPENHANDS_CSV', 'analysis_outputs/openhands_prs_2024-03-13_to_2025-06-11.csv')
    oh_commits_csv = os.getenv('OH_MONTHLY_COMMITS', 'analysis_outputs/openhands_monthly_commits_2024-03-13_to_2025-06-11.csv')
    aidev_parquet = os.getenv('AIDEV_FULL_PARQUET', 'all_pull_request.parquet')

    # Load OpenHands PRs
    oh_prs = pd.read_csv(openhands_csv)
    oh_prs['created_at'] = pd.to_datetime(oh_prs['created_at'])
    oh_prs['merged_at'] = pd.to_datetime(oh_prs['merged_at'])
    oh_prs['closed_at'] = pd.to_datetime(oh_prs['closed_at'])
    oh_prs['dataset'] = 'OpenHands'
    oh_prs['repo'] = 'OpenHands'
    
    # Load OpenHands commit data
    oh_commits = pd.read_csv(oh_commits_csv)
    oh_avg_commits = oh_commits['avg_commits_per_pr'].mean()
    
    # Load AIDev PRs (FULL dataset)
    aidev_full = pd.read_parquet(aidev_parquet)
    aidev_full['created_at'] = pd.to_datetime(aidev_full['created_at'], utc=True)
    aidev_full['merged_at'] = pd.to_datetime(aidev_full['merged_at'], utc=True)
    aidev_full['closed_at'] = pd.to_datetime(aidev_full['closed_at'], utc=True)
    aidev_full['dataset'] = 'AI-Dev Full'
    aidev_full['repo'] = aidev_full['repo_url'].str.split('/').str[-1]
    
    # Time filter AIDev
    start_date = pd.to_datetime('2024-03-13', utc=True)
    end_date = pd.to_datetime('2025-06-11', utc=True)
    aidev_time = aidev_full[
        (aidev_full['created_at'] >= start_date) & 
        (aidev_full['created_at'] <= end_date)
    ].copy()
    aidev_time['dataset'] = 'AI-Dev Time'
    
    # Load commit count data for AI-Dev repos
    try:
        aidev_full_commits = pd.read_csv('analysis_outputs/aidev_full_top5_commit_size.csv')
        aidev_full_commits['repo'] = aidev_full_commits['repo_name'].str.split('/').str[-1]
        commit_lookup_full = aidev_full_commits.set_index('repo')['avg_commits_per_pr'].to_dict()
        
        aidev_time_commits = pd.read_csv('analysis_outputs/aidev_time_top5_commit_size.csv')
        aidev_time_commits['repo'] = aidev_time_commits['repo_name'].str.split('/').str[-1]
        commit_lookup_time = aidev_time_commits.set_index('repo')['avg_commits_per_pr'].to_dict()
    except:
        print("Warning: Could not load commit size data")
        commit_lookup_full = {}
        commit_lookup_time = {}
    
    # Calculate time to merge (in hours)
    oh_prs['time_to_merge_hours'] = (oh_prs['merged_at'] - oh_prs['created_at']).dt.total_seconds() / 3600
    aidev_full['time_to_merge_hours'] = (aidev_full['merged_at'] - aidev_full['created_at']).dt.total_seconds() / 3600
    aidev_time['time_to_merge_hours'] = (aidev_time['merged_at'] - aidev_time['created_at']).dt.total_seconds() / 3600
    
    # Calculate merge status
    oh_prs['is_merged'] = oh_prs['merged_at'].notna()
    aidev_full['is_merged'] = aidev_full['merged_at'].notna()
    aidev_time['is_merged'] = aidev_time['merged_at'].notna()
    
    print(f"OpenHands PRs: {len(oh_prs):,}")
    print(f"  Merged: {oh_prs['is_merged'].sum():,} ({oh_prs['is_merged'].mean()*100:.2f}%)")
    print(f"  Avg Commits/PR: {oh_avg_commits:.2f}")
    print(f"AI-Dev Full PRs: {len(aidev_full):,}")
    print(f"  Merged: {aidev_full['is_merged'].sum():,} ({aidev_full['is_merged'].mean()*100:.2f}%)")
    print(f"AI-Dev Time PRs: {len(aidev_time):,}")
    print(f"  Merged: {aidev_time['is_merged'].sum():,} ({aidev_time['is_merged'].mean()*100:.2f}%)")
    
    return oh_prs, aidev_full, aidev_time, oh_avg_commits, commit_lookup_full, commit_lookup_time

def calculate_merge_rates(oh_prs, aidev_full, aidev_time, oh_avg_commits, commit_lookup_full, commit_lookup_time):
    """Calculate merge rates overall and by repository"""
    print("\n" + "="*80)
    print("MERGE RATE ANALYSIS")
    print("="*80)
    
    # Overall merge rates
    oh_merge_rate = oh_prs['is_merged'].mean() * 100
    aidev_full_merge_rate = aidev_full['is_merged'].mean() * 100
    aidev_time_merge_rate = aidev_time['is_merged'].mean() * 100
    
    print(f"\nOverall Merge Rates:")
    print(f"  OpenHands: {oh_merge_rate:.2f}% ({oh_prs['is_merged'].sum()}/{len(oh_prs)})")
    print(f"  AI-Dev Full: {aidev_full_merge_rate:.2f}% ({aidev_full['is_merged'].sum()}/{len(aidev_full)})")
    print(f"  AI-Dev Time: {aidev_time_merge_rate:.2f}% ({aidev_time['is_merged'].sum()}/{len(aidev_time)})")
    
    # Top repos from AI-Dev Full
    print(f"\nTop 5 AI-Dev Full Repositories by PR Count:")
    top_repos_full = ['mochi', 'codeforces', 'AGI-Alpha-Agent-v0', 'MVP-website', 'TonPlaygramWebApp']
    
    repo_stats = []
    for repo in top_repos_full:
        repo_prs = aidev_full[aidev_full['repo'] == repo]
        if len(repo_prs) == 0:
            continue
        merge_rate = repo_prs['is_merged'].mean() * 100
        merged_count = repo_prs['is_merged'].sum()
        total_count = len(repo_prs)
        avg_commits_pr = commit_lookup_full.get(repo, np.nan)
        
        # Chi-square test vs OpenHands
        oh_merged = oh_prs['is_merged'].sum()
        oh_total = len(oh_prs)
        contingency = np.array([
            [oh_merged, oh_total - oh_merged],
            [merged_count, total_count - merged_count]
        ])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        commits_str = f"{avg_commits_pr:.2f}" if not np.isnan(avg_commits_pr) else "N/A"
        print(f"  {repo}: {merge_rate:.2f}% ({merged_count}/{total_count}), commits/PR: {commits_str}, χ² = {chi2:.1f}, p = {p_value:.2e}")
        
        repo_stats.append({
            'dataset': 'AI-Dev Full',
            'repo': repo,
            'merge_rate': merge_rate,
            'merged_count': merged_count,
            'total_count': total_count,
            'avg_commits_pr': avg_commits_pr,
            'chi2': chi2,
            'p_value': p_value
        })
    
    # Top repos from AI-Dev Time
    print(f"\nTop 5 AI-Dev Time Repositories by PR Count:")
    top_repos_time = ['AGI-Alpha-Agent-v0', 'glimpser', 'ai', 'LemmingsJS-MIDI', 'alfe-ai']
    
    for repo in top_repos_time:
        repo_prs = aidev_time[aidev_time['repo'] == repo]
        if len(repo_prs) == 0:
            continue
        merge_rate = repo_prs['is_merged'].mean() * 100
        merged_count = repo_prs['is_merged'].sum()
        total_count = len(repo_prs)
        avg_commits_pr = commit_lookup_time.get(repo, np.nan)
        
        # Chi-square test vs OpenHands
        oh_merged = oh_prs['is_merged'].sum()
        oh_total = len(oh_prs)
        contingency = np.array([
            [oh_merged, oh_total - oh_merged],
            [merged_count, total_count - merged_count]
        ])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        commits_str = f"{avg_commits_pr:.2f}" if not np.isnan(avg_commits_pr) else "N/A"
        print(f"  {repo}: {merge_rate:.2f}% ({merged_count}/{total_count}), commits/PR: {commits_str}, χ² = {chi2:.1f}, p = {p_value:.2e}")
        
        repo_stats.append({
            'dataset': 'AI-Dev Time',
            'repo': repo,
            'merge_rate': merge_rate,
            'merged_count': merged_count,
            'total_count': total_count,
            'avg_commits_pr': avg_commits_pr,
            'chi2': chi2,
            'p_value': p_value
        })
    
    return pd.DataFrame(repo_stats), oh_avg_commits

def analyze_time_to_merge(oh_prs, aidev_full, aidev_time):
    """Analyze time to merge distributions"""
    print("\n" + "="*80)
    print("TIME TO MERGE ANALYSIS")
    print("="*80)
    
    # Filter only merged PRs
    oh_merged = oh_prs[oh_prs['is_merged'] & (oh_prs['time_to_merge_hours'] >= 0)].copy()
    aidev_full_merged = aidev_full[aidev_full['is_merged'] & (aidev_full['time_to_merge_hours'] >= 0)].copy()
    aidev_time_merged = aidev_time[aidev_time['is_merged'] & (aidev_time['time_to_merge_hours'] >= 0)].copy()
    
    # Filter outliers (cap at 99th percentile for cleaner visualization)
    oh_p99 = oh_merged['time_to_merge_hours'].quantile(0.99)
    aidev_full_p99 = aidev_full_merged['time_to_merge_hours'].quantile(0.99)
    aidev_time_p99 = aidev_time_merged['time_to_merge_hours'].quantile(0.99)
    
    print(f"\nOpenHands Time to Merge (hours):")
    print(f"  Mean: {oh_merged['time_to_merge_hours'].mean():.2f}")
    print(f"  Median: {oh_merged['time_to_merge_hours'].median():.2f}")
    print(f"  Std: {oh_merged['time_to_merge_hours'].std():.2f}")
    print(f"  Min: {oh_merged['time_to_merge_hours'].min():.2f}")
    print(f"  25th percentile: {oh_merged['time_to_merge_hours'].quantile(0.25):.2f}")
    print(f"  75th percentile: {oh_merged['time_to_merge_hours'].quantile(0.75):.2f}")
    print(f"  95th percentile: {oh_merged['time_to_merge_hours'].quantile(0.95):.2f}")
    print(f"  99th percentile: {oh_p99:.2f}")
    print(f"  Max: {oh_merged['time_to_merge_hours'].max():.2f}")
    
    print(f"\nAI-Dev Full Time to Merge (hours):")
    print(f"  Mean: {aidev_full_merged['time_to_merge_hours'].mean():.2f}")
    print(f"  Median: {aidev_full_merged['time_to_merge_hours'].median():.2f}")
    print(f"  Std: {aidev_full_merged['time_to_merge_hours'].std():.2f}")
    print(f"  Min: {aidev_full_merged['time_to_merge_hours'].min():.2f}")
    print(f"  25th percentile: {aidev_full_merged['time_to_merge_hours'].quantile(0.25):.2f}")
    print(f"  75th percentile: {aidev_full_merged['time_to_merge_hours'].quantile(0.75):.2f}")
    print(f"  95th percentile: {aidev_full_merged['time_to_merge_hours'].quantile(0.95):.2f}")
    print(f"  99th percentile: {aidev_full_p99:.2f}")
    print(f"  Max: {aidev_full_merged['time_to_merge_hours'].max():.2f}")
    
    print(f"\nAI-Dev Time Time to Merge (hours):")
    print(f"  Mean: {aidev_time_merged['time_to_merge_hours'].mean():.2f}")
    print(f"  Median: {aidev_time_merged['time_to_merge_hours'].median():.2f}")
    print(f"  Std: {aidev_time_merged['time_to_merge_hours'].std():.2f}")
    print(f"  Min: {aidev_time_merged['time_to_merge_hours'].min():.2f}")
    print(f"  25th percentile: {aidev_time_merged['time_to_merge_hours'].quantile(0.25):.2f}")
    print(f"  75th percentile: {aidev_time_merged['time_to_merge_hours'].quantile(0.75):.2f}")
    print(f"  95th percentile: {aidev_time_merged['time_to_merge_hours'].quantile(0.95):.2f}")
    print(f"  99th percentile: {aidev_time_p99:.2f}")
    print(f"  Max: {aidev_time_merged['time_to_merge_hours'].max():.2f}")
    
    # Statistical tests
    print(f"\nStatistical Tests (OpenHands vs AI-Dev Full):")
    
    # Mann-Whitney U test
    u_stat_full, u_p_full = stats.mannwhitneyu(oh_merged['time_to_merge_hours'], 
                                      aidev_full_merged['time_to_merge_hours'], 
                                      alternative='two-sided')
    print(f"  Mann-Whitney U: U = {u_stat_full:.2f}, p = {u_p_full:.2e}")
    print(f"  Interpretation: {'SIGNIFICANT' if u_p_full < 0.001 else 'Not significant'} difference")
    
    # Effect size (Cohen's d)
    pooled_std_full = np.sqrt((oh_merged['time_to_merge_hours'].std()**2 + 
                          aidev_full_merged['time_to_merge_hours'].std()**2) / 2)
    cohens_d_full = (oh_merged['time_to_merge_hours'].mean() - 
                aidev_full_merged['time_to_merge_hours'].mean()) / pooled_std_full
    print(f"  Cohen's d: {cohens_d_full:.4f}")
    
    print(f"\nStatistical Tests (OpenHands vs AI-Dev Time):")
    
    # Mann-Whitney U test
    u_stat_time, u_p_time = stats.mannwhitneyu(oh_merged['time_to_merge_hours'], 
                                      aidev_time_merged['time_to_merge_hours'], 
                                      alternative='two-sided')
    print(f"  Mann-Whitney U: U = {u_stat_time:.2f}, p = {u_p_time:.2e}")
    print(f"  Interpretation: {'SIGNIFICANT' if u_p_time < 0.001 else 'Not significant'} difference")
    
    # Effect size (Cohen's d)
    pooled_std_time = np.sqrt((oh_merged['time_to_merge_hours'].std()**2 + 
                          aidev_time_merged['time_to_merge_hours'].std()**2) / 2)
    cohens_d_time = (oh_merged['time_to_merge_hours'].mean() - 
                aidev_time_merged['time_to_merge_hours'].mean()) / pooled_std_time
    print(f"  Cohen's d: {cohens_d_time:.4f}")
    
    # Convert to days for reporting
    oh_merged['time_to_merge_days'] = oh_merged['time_to_merge_hours'] / 24
    aidev_full_merged['time_to_merge_days'] = aidev_full_merged['time_to_merge_hours'] / 24
    aidev_time_merged['time_to_merge_days'] = aidev_time_merged['time_to_merge_hours'] / 24
    
    return oh_merged, aidev_full_merged, aidev_time_merged, {
        'u_stat_full': u_stat_full,
        'u_p_full': u_p_full,
        'cohens_d_full': cohens_d_full,
        'u_stat_time': u_stat_time,
        'u_p_time': u_p_time,
        'cohens_d_time': cohens_d_time
    }

def create_visualizations(oh_prs, aidev_df, oh_merged, aidev_merged, repo_stats, stats_results):
    """Create comprehensive time to merge visualizations"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # 1. Merge Rate Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    datasets = ['OpenHands', 'AI-Dev']
    merge_rates = [
        oh_prs['is_merged'].mean() * 100,
        aidev_df['is_merged'].mean() * 100
    ]
    colors = ['#e74c3c', '#3498db']
    bars = ax1.bar(datasets, merge_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Merge Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Merge Rates', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rate in zip(bars, merge_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Top 10 Repos Merge Rates
    ax2 = fig.add_subplot(gs[0, 1:])
    top10 = repo_stats.head(10).sort_values('merge_rate', ascending=True)
    y_pos = np.arange(len(top10))
    
    # Color bars by significance
    colors = ['#e74c3c' if p < 0.001 else '#95a5a6' for p in top10['p_value']]
    bars = ax2.barh(y_pos, top10['merge_rate'], color=colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top10['repo'], fontsize=10)
    ax2.set_xlabel('Merge Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Top 10 AI-Dev Repositories Merge Rates\n(Red = Significant difference from OpenHands, p < 0.001)', 
                  fontsize=12, fontweight='bold')
    ax2.axvline(merge_rates[0], color='#9b59b6', linestyle='--', linewidth=2, label=f'OpenHands: {merge_rates[0]:.1f}%')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, top10['merge_rate'])):
        ax2.text(rate + 1, bar.get_y() + bar.get_height()/2.,
                f'{rate:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    # 3. Time to Merge Distribution (Box Plot)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Cap outliers for visualization
    oh_plot = oh_merged[oh_merged['time_to_merge_hours'] <= oh_merged['time_to_merge_hours'].quantile(0.95)]
    aidev_plot = aidev_merged[aidev_merged['time_to_merge_hours'] <= aidev_merged['time_to_merge_hours'].quantile(0.95)]
    
    combined_time = pd.concat([
        oh_plot[['dataset', 'time_to_merge_hours']],
        aidev_plot[['dataset', 'time_to_merge_hours']]
    ])
    
    sns.boxplot(data=combined_time, y='dataset', x='time_to_merge_hours', ax=ax3, palette=['#e74c3c', '#3498db'])
    ax3.set_xlabel('Time to Merge (hours)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Dataset', fontsize=11, fontweight='bold')
    ax3.set_title('Time to Merge Distribution\n(95th percentile cap)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Time to Merge Distribution (Violin Plot)
    ax4 = fig.add_subplot(gs[1, 1])
    sns.violinplot(data=combined_time, y='dataset', x='time_to_merge_hours', ax=ax4, palette=['#e74c3c', '#3498db'])
    ax4.set_xlabel('Time to Merge (hours)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Dataset', fontsize=11, fontweight='bold')
    ax4.set_title('Time to Merge Distribution (Violin)\n(95th percentile cap)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Time to Merge Histogram (Log Scale)
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Use logarithmic bins
    min_val = min(oh_merged['time_to_merge_hours'].min(), aidev_merged['time_to_merge_hours'].min())
    max_val = max(oh_merged['time_to_merge_hours'].max(), aidev_merged['time_to_merge_hours'].max())
    bins = np.logspace(np.log10(max(min_val, 0.01)), np.log10(max_val), 40)
    
    ax5.hist(oh_merged['time_to_merge_hours'], bins=bins, alpha=0.6, label='OpenHands', color='#e74c3c', edgecolor='black')
    ax5.hist(aidev_merged['time_to_merge_hours'], bins=bins, alpha=0.6, label='AI-Dev', color='#3498db', edgecolor='black')
    ax5.set_xscale('log')
    ax5.set_xlabel('Time to Merge (hours, log scale)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('Time to Merge Distribution\n(Logarithmic Scale)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative Distribution Function
    ax6 = fig.add_subplot(gs[2, 0])
    
    oh_sorted = np.sort(oh_merged['time_to_merge_hours'])
    aidev_sorted = np.sort(aidev_merged['time_to_merge_hours'])
    oh_cdf = np.arange(1, len(oh_sorted) + 1) / len(oh_sorted)
    aidev_cdf = np.arange(1, len(aidev_sorted) + 1) / len(aidev_sorted)
    
    ax6.plot(oh_sorted, oh_cdf, linewidth=2, label='OpenHands', color='#e74c3c')
    ax6.plot(aidev_sorted, aidev_cdf, linewidth=2, label='AI-Dev', color='#3498db')
    ax6.set_xlabel('Time to Merge (hours)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax6.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax6.set_xlim(0, 200)  # Cap at 200 hours for readability
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # 7. Time to Merge by Quantiles
    ax7 = fig.add_subplot(gs[2, 1])
    quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    oh_quantiles = [oh_merged['time_to_merge_hours'].quantile(q) for q in quantiles]
    aidev_quantiles = [aidev_merged['time_to_merge_hours'].quantile(q) for q in quantiles]
    
    x = np.arange(len(quantiles))
    width = 0.35
    
    ax7.bar(x - width/2, oh_quantiles, width, label='OpenHands', color='#e74c3c', alpha=0.7, edgecolor='black')
    ax7.bar(x + width/2, aidev_quantiles, width, label='AI-Dev', color='#3498db', alpha=0.7, edgecolor='black')
    ax7.set_xticks(x)
    ax7.set_xticklabels([f'{int(q*100)}%' for q in quantiles], fontsize=10)
    ax7.set_ylabel('Time to Merge (hours)', fontsize=11, fontweight='bold')
    ax7.set_xlabel('Quantile', fontsize=11, fontweight='bold')
    ax7.set_title('Time to Merge by Quantiles', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. PR Lifecycle States
    ax8 = fig.add_subplot(gs[2, 2])
    
    oh_states = pd.Series({
        'Merged': oh_prs['is_merged'].sum(),
        'Closed (not merged)': (~oh_prs['is_merged'] & oh_prs['closed_at'].notna()).sum(),
        'Open': oh_prs['state'].eq('open').sum()
    })
    
    aidev_states = pd.Series({
        'Merged': aidev_df['is_merged'].sum(),
        'Closed (not merged)': (~aidev_df['is_merged'] & aidev_df['closed_at'].notna()).sum(),
        'Open': aidev_df['state'].eq('open').sum()
    })
    
    oh_states_pct = oh_states / oh_states.sum() * 100
    aidev_states_pct = aidev_states / aidev_states.sum() * 100
    
    x = np.arange(len(oh_states))
    width = 0.35
    
    ax8.bar(x - width/2, oh_states_pct, width, label='OpenHands', color='#e74c3c', alpha=0.7, edgecolor='black')
    ax8.bar(x + width/2, aidev_states_pct, width, label='AI-Dev', color='#3498db', alpha=0.7, edgecolor='black')
    ax8.set_xticks(x)
    ax8.set_xticklabels(oh_states.index, fontsize=10, rotation=15, ha='right')
    ax8.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax8.set_title('PR Lifecycle States', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (oh_val, ai_val) in enumerate(zip(oh_states_pct, aidev_states_pct)):
        ax8.text(i - width/2, oh_val + 1, f'{oh_val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax8.text(i + width/2, ai_val + 1, f'{ai_val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 9. Time to Merge in Days (Box Plot with outliers removed)
    ax9 = fig.add_subplot(gs[3, 0])
    
    # Cap at 30 days for cleaner visualization
    oh_days_plot = oh_merged[oh_merged['time_to_merge_days'] <= 30]
    aidev_days_plot = aidev_merged[aidev_merged['time_to_merge_days'] <= 30]
    
    combined_days = pd.concat([
        oh_days_plot[['dataset', 'time_to_merge_days']],
        aidev_days_plot[['dataset', 'time_to_merge_days']]
    ])
    
    sns.boxplot(data=combined_days, y='dataset', x='time_to_merge_days', ax=ax9, palette=['#e74c3c', '#3498db'])
    ax9.set_xlabel('Time to Merge (days)', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Dataset', fontsize=11, fontweight='bold')
    ax9.set_title('Time to Merge Distribution (Days)\n(Capped at 30 days)', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='x')
    
    # 10. Fast Merges (< 1 hour)
    ax10 = fig.add_subplot(gs[3, 1])
    
    oh_fast = (oh_merged['time_to_merge_hours'] < 1).sum() / len(oh_merged) * 100
    oh_medium = ((oh_merged['time_to_merge_hours'] >= 1) & (oh_merged['time_to_merge_hours'] < 24)).sum() / len(oh_merged) * 100
    oh_slow = (oh_merged['time_to_merge_hours'] >= 24).sum() / len(oh_merged) * 100
    
    aidev_fast = (aidev_merged['time_to_merge_hours'] < 1).sum() / len(aidev_merged) * 100
    aidev_medium = ((aidev_merged['time_to_merge_hours'] >= 1) & (aidev_merged['time_to_merge_hours'] < 24)).sum() / len(aidev_merged) * 100
    aidev_slow = (aidev_merged['time_to_merge_hours'] >= 24).sum() / len(aidev_merged) * 100
    
    categories = ['< 1 hour', '1-24 hours', '≥ 24 hours']
    oh_values = [oh_fast, oh_medium, oh_slow]
    aidev_values = [aidev_fast, aidev_medium, aidev_slow]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax10.bar(x - width/2, oh_values, width, label='OpenHands', color='#e74c3c', alpha=0.7, edgecolor='black')
    ax10.bar(x + width/2, aidev_values, width, label='AI-Dev', color='#3498db', alpha=0.7, edgecolor='black')
    ax10.set_xticks(x)
    ax10.set_xticklabels(categories, fontsize=10)
    ax10.set_ylabel('Percentage of Merged PRs (%)', fontsize=11, fontweight='bold')
    ax10.set_title('Merge Speed Categories', fontsize=12, fontweight='bold')
    ax10.legend(fontsize=10)
    ax10.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (oh_val, ai_val) in enumerate(zip(oh_values, aidev_values)):
        ax10.text(i - width/2, oh_val + 1, f'{oh_val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax10.text(i + width/2, ai_val + 1, f'{ai_val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 11. Statistical Results Text
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.axis('off')
    
    stats_text = f"""
Statistical Test Results
{'='*35}

Time to Merge Comparison:

OpenHands:
  Median: {oh_merged['time_to_merge_hours'].median():.2f} hours
  Mean: {oh_merged['time_to_merge_hours'].mean():.2f} hours
  
AI-Dev:
  Median: {aidev_merged['time_to_merge_hours'].median():.2f} hours
  Mean: {aidev_merged['time_to_merge_hours'].mean():.2f} hours

Mann-Whitney U Test:
  U = {stats_results['u_stat']:.2f}
  p = {stats_results['u_p']:.2e}
  {'✓ SIGNIFICANT' if stats_results['u_p'] < 0.001 else '✗ Not significant'}

Effect Size:
  Cohen's d = {stats_results['cohens_d']:.4f}
"""
    
    ax11.text(0.05, 0.95, stats_text, transform=ax11.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('RQ3: Time to Merge and PR Lifecycle Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    output_file = 'analysis_outputs/rq3_time_to_merge_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def save_summary_data(oh_prs, aidev_df, oh_merged, aidev_merged, repo_stats):
    """Save summary data to CSV"""
    # Merge rate summary
    merge_summary = pd.DataFrame([
        {
            'dataset': 'OpenHands',
            'total_prs': len(oh_prs),
            'merged_prs': oh_prs['is_merged'].sum(),
            'merge_rate': oh_prs['is_merged'].mean() * 100
        },
        {
            'dataset': 'AI-Dev',
            'total_prs': len(aidev_df),
            'merged_prs': aidev_df['is_merged'].sum(),
            'merge_rate': aidev_df['is_merged'].mean() * 100
        }
    ])
    merge_summary.to_csv('analysis_outputs/rq3_merge_rates_summary.csv', index=False)
    
    # Repo-level stats
    repo_stats.to_csv('analysis_outputs/rq3_repo_merge_stats.csv', index=False)
    
    # Time to merge summary
    time_summary = pd.DataFrame([
        {
            'dataset': 'OpenHands',
            'mean_hours': oh_merged['time_to_merge_hours'].mean(),
            'median_hours': oh_merged['time_to_merge_hours'].median(),
            'std_hours': oh_merged['time_to_merge_hours'].std(),
            'p25_hours': oh_merged['time_to_merge_hours'].quantile(0.25),
            'p75_hours': oh_merged['time_to_merge_hours'].quantile(0.75),
            'p95_hours': oh_merged['time_to_merge_hours'].quantile(0.95),
            'fast_pct': (oh_merged['time_to_merge_hours'] < 1).sum() / len(oh_merged) * 100,
            'medium_pct': ((oh_merged['time_to_merge_hours'] >= 1) & (oh_merged['time_to_merge_hours'] < 24)).sum() / len(oh_merged) * 100,
            'slow_pct': (oh_merged['time_to_merge_hours'] >= 24).sum() / len(oh_merged) * 100
        },
        {
            'dataset': 'AI-Dev',
            'mean_hours': aidev_merged['time_to_merge_hours'].mean(),
            'median_hours': aidev_merged['time_to_merge_hours'].median(),
            'std_hours': aidev_merged['time_to_merge_hours'].std(),
            'p25_hours': aidev_merged['time_to_merge_hours'].quantile(0.25),
            'p75_hours': aidev_merged['time_to_merge_hours'].quantile(0.75),
            'p95_hours': aidev_merged['time_to_merge_hours'].quantile(0.95),
            'fast_pct': (aidev_merged['time_to_merge_hours'] < 1).sum() / len(aidev_merged) * 100,
            'medium_pct': ((aidev_merged['time_to_merge_hours'] >= 1) & (aidev_merged['time_to_merge_hours'] < 24)).sum() / len(aidev_merged) * 100,
            'slow_pct': (aidev_merged['time_to_merge_hours'] >= 24).sum() / len(aidev_merged) * 100
        }
    ])
    time_summary.to_csv('analysis_outputs/rq3_time_to_merge_summary.csv', index=False)
    
    print("\n✓ Saved summary CSV files")

def main():
    print("="*80)
    print("RQ3: TIME TO MERGE AND PR LIFECYCLE ANALYSIS")
    print("="*80)
    
    # Load data
    oh_prs, aidev_full, aidev_time, oh_avg_commits, commit_lookup_full, commit_lookup_time = load_and_prepare_data()
    
    # Merge rate analysis
    repo_stats, oh_avg_commits_val = calculate_merge_rates(oh_prs, aidev_full, aidev_time, oh_avg_commits, commit_lookup_full, commit_lookup_time)
    
    # Time to merge analysis
    oh_merged, aidev_full_merged, aidev_time_merged, stats_results = analyze_time_to_merge(oh_prs, aidev_full, aidev_time)
    
    print("\n" + "="*80)
    print("RQ3 ANALYSIS COMPLETE")
    print("="*80)
    
    print("\n" + "="*80)
    print("RQ3 ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
