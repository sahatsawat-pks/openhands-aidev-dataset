"""
RQ2: Repository-Level Temporal Activity Patterns
Includes BOTH daily and monthly levels, burstiness coefficient, and correct repos
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_burstiness(counts):
    """Calculate burstiness coefficient (Goh & Barabási, 2008)"""
    mean = counts.mean()
    std = counts.std()
    if mean + std == 0:
        return 0
    return (std - mean) / (std + mean)

def load_data():
    """Load PR data from both datasets"""
    print("Loading datasets...")
    
    # Load OpenHands PRs
    oh_prs = pd.read_csv('analysis_outputs/openhands_prs_2024-03-13_to_2025-06-11.csv')
    oh_prs['created_at'] = pd.to_datetime(oh_prs['created_at'])
    oh_prs['repo'] = 'OpenHands'
    
    # Load OpenHands commit data
    oh_commits = pd.read_csv('analysis_outputs/openhands_monthly_commits_2024-03-13_to_2025-06-11.csv')
    
    # Load AIDev PRs
    aidev_df = pd.read_parquet('all_pull_request.parquet')
    aidev_df['created_at'] = pd.to_datetime(aidev_df['created_at'], utc=True)
    
    # Time filter
    start_date = pd.to_datetime('2024-03-13', utc=True)
    end_date = pd.to_datetime('2025-06-11', utc=True)
    aidev_time = aidev_df[
        (aidev_df['created_at'] >= start_date) & 
        (aidev_df['created_at'] <= end_date)
    ].copy()
    
    # Extract repo name
    aidev_df['repo'] = aidev_df['repo_url'].str.split('/').str[-1]
    aidev_time['repo'] = aidev_time['repo_url'].str.split('/').str[-1]
    
    # Load commit size data for AI-Dev repos
    aidev_full_commits = pd.read_csv('analysis_outputs/aidev_full_top5_commit_size.csv')
    aidev_time_commits = pd.read_csv('analysis_outputs/aidev_time_top5_commit_size.csv')
    
    # Create commit lookup dictionaries
    aidev_full_commits['repo'] = aidev_full_commits['repo_name'].str.split('/').str[-1]
    aidev_time_commits['repo'] = aidev_time_commits['repo_name'].str.split('/').str[-1]
    
    commit_lookup = {
        'full': {
            'avg_per_pr': aidev_full_commits.set_index('repo')['avg_commits_per_pr'].to_dict(),
            'total': aidev_full_commits.set_index('repo')['total_commits'].to_dict()
        },
        'time': {
            'avg_per_pr': aidev_time_commits.set_index('repo')['avg_commits_per_pr'].to_dict(),
            'total': aidev_time_commits.set_index('repo')['total_commits'].to_dict()
        }
    }
    
    return oh_prs, aidev_df, aidev_time, oh_commits, commit_lookup

def calculate_repo_metrics(prs_df, total_commits=None):
    """Calculate daily and monthly activity metrics for a repository"""
    # Daily counts
    daily_counts = prs_df.groupby(prs_df['created_at'].dt.date).size()
    num_days = len(daily_counts)
    
    # Monthly counts
    prs_df['year_month'] = prs_df['created_at'].dt.to_period('M')
    monthly_counts = prs_df.groupby('year_month').size()
    num_months = len(monthly_counts)
    
    # Calculate burstiness
    daily_burstiness = calculate_burstiness(daily_counts)
    monthly_burstiness = calculate_burstiness(monthly_counts)
    
    # Calculate commit rates if total_commits provided
    commits_per_day = total_commits / num_days if total_commits is not None and num_days > 0 else np.nan
    commits_per_month = total_commits / num_months if total_commits is not None and num_months > 0 else np.nan
    
    return {
        'daily_mean': daily_counts.mean(),
        'daily_std': daily_counts.std(),
        'daily_counts': daily_counts.values,
        'daily_burstiness': daily_burstiness,
        'monthly_mean': monthly_counts.mean(),
        'monthly_std': monthly_counts.std(),
        'monthly_counts': monthly_counts.values,
        'monthly_burstiness': monthly_burstiness,
        'total_prs': len(prs_df),
        'commits_per_day': commits_per_day,
        'commits_per_month': commits_per_month
    }

def compare_repositories(oh_prs, aidev_full, aidev_time, oh_commits, commit_lookup):
    """Compare daily and monthly activity across repositories"""
    print("\n" + "="*80)
    print("REPOSITORY-LEVEL TEMPORAL ACTIVITY COMPARISON")
    print("="*80)
    
    # Calculate OpenHands baseline
    oh_total_commits = oh_commits['total_commits'].sum()
    oh_stats = calculate_repo_metrics(oh_prs, oh_total_commits)
    oh_daily = oh_stats['daily_counts']
    oh_monthly = oh_stats['monthly_counts']
    
    # Calculate OpenHands avg commits per PR
    oh_avg_commits_pr = oh_commits['avg_commits_per_pr'].mean()
    
    print(f"\nOpenHands:")
    print(f"  Daily: {oh_stats['daily_mean']:.2f} PRs/day, {oh_stats['commits_per_day']:.2f} Commits/day, B={oh_stats['daily_burstiness']:.4f}")
    print(f"  Monthly: {oh_stats['monthly_mean']:.2f} PRs/month, {oh_stats['commits_per_month']:.2f} Commits/month, B={oh_stats['monthly_burstiness']:.4f}")
    print(f"  Avg Commits/PR: {oh_avg_commits_pr:.2f}")
    
    results = []
    
    # Define correct top 5 for AIDev-All
    print("\n--- AIDev-All (Top 5) ---")
    
    # Get top repos excluding Poker_Analyzer
    top_repos_counts = aidev_full['repo'].value_counts()
    
    # Manually select correct repos
    correct_repos_all = ['mochi', 'codeforces', 'AGI-Alpha-Agent-v0', 'MVP-website', 'TonPlaygramWebApp']
    
    for repo in correct_repos_all:
        if repo not in top_repos_counts.index:
            print(f"Warning: {repo} not found in AIDev-All")
            continue
            
        repo_prs = aidev_full[aidev_full['repo'] == repo]
        total_commits = commit_lookup['full']['total'].get(repo, None)
        repo_stats = calculate_repo_metrics(repo_prs, total_commits)
        
        # Get avg commits per PR
        avg_commits_pr = commit_lookup['full']['avg_per_pr'].get(repo, np.nan)
        
        # Mann-Whitney U test - daily
        u_daily, p_daily = stats.mannwhitneyu(oh_daily, repo_stats['daily_counts'], alternative='two-sided')
        
        # Effect size (Cohen's d) - daily
        pooled_std_daily = np.sqrt((oh_stats['daily_std']**2 + repo_stats['daily_std']**2) / 2)
        cohens_d_daily = (oh_stats['daily_mean'] - repo_stats['daily_mean']) / pooled_std_daily if pooled_std_daily > 0 else 0
        
        # Mann-Whitney U test - monthly
        u_monthly, p_monthly = stats.mannwhitneyu(oh_monthly, repo_stats['monthly_counts'], alternative='two-sided')
        
        # Effect size (Cohen's d) - monthly
        pooled_std_monthly = np.sqrt((oh_stats['monthly_std']**2 + repo_stats['monthly_std']**2) / 2)
        cohens_d_monthly = (oh_stats['monthly_mean'] - repo_stats['monthly_mean']) / pooled_std_monthly if pooled_std_monthly > 0 else 0
        
        print(f"{repo}:")
        print(f"  Daily: {repo_stats['daily_mean']:.2f} PRs/day, {repo_stats['commits_per_day']:.2f} Commits/day, B={repo_stats['daily_burstiness']:.4f}, d={cohens_d_daily:.4f}")
        print(f"  Monthly: {repo_stats['monthly_mean']:.2f} PRs/month, {repo_stats['commits_per_month']:.2f} Commits/month, B={repo_stats['monthly_burstiness']:.4f}, d={cohens_d_monthly:.4f}")
        print(f"  Avg Commits/PR: {avg_commits_pr:.2f}")
        
        results.append({
            'dataset': 'AIDev-All',
            'repo': repo,
            'daily_mean': repo_stats['daily_mean'],
            'daily_burstiness': repo_stats['daily_burstiness'],
            'cohens_d_daily': cohens_d_daily,
            'p_value_daily': p_daily,
            'monthly_mean': repo_stats['monthly_mean'],
            'monthly_burstiness': repo_stats['monthly_burstiness'],
            'cohens_d_monthly': cohens_d_monthly,
            'p_value_monthly': p_monthly,
            'total_prs': len(repo_prs),
            'avg_commits_pr': avg_commits_pr,
            'commits_per_day': repo_stats['commits_per_day'],
            'commits_per_month': repo_stats['commits_per_month']
        })
    
    # AIDev-Time (top 5)
    print("\n--- AIDev-Time (Top 5) ---")
    
    correct_repos_time = ['AGI-Alpha-Agent-v0', 'glimpser', 'ai', 'LemmingsJS-MIDI', 'alfe-ai']
    
    for repo in correct_repos_time:
        repo_prs = aidev_time[aidev_time['repo'] == repo]
        if len(repo_prs) == 0:
            print(f"Warning: {repo} not found in AIDev-Time")
            continue
            
        total_commits = commit_lookup['time']['total'].get(repo, None)
        repo_stats = calculate_repo_metrics(repo_prs, total_commits)
        
        # Get avg commits per PR
        avg_commits_pr = commit_lookup['time']['avg_per_pr'].get(repo, np.nan)
        
        # Mann-Whitney U test - daily
        u_daily, p_daily = stats.mannwhitneyu(oh_daily, repo_stats['daily_counts'], alternative='two-sided')
        
        # Effect size (Cohen's d) - daily
        pooled_std_daily = np.sqrt((oh_stats['daily_std']**2 + repo_stats['daily_std']**2) / 2)
        cohens_d_daily = (oh_stats['daily_mean'] - repo_stats['daily_mean']) / pooled_std_daily if pooled_std_daily > 0 else 0
        
        # Mann-Whitney U test - monthly
        u_monthly, p_monthly = stats.mannwhitneyu(oh_monthly, repo_stats['monthly_counts'], alternative='two-sided')
        
        # Effect size (Cohen's d) - monthly
        pooled_std_monthly = np.sqrt((oh_stats['monthly_std']**2 + repo_stats['monthly_std']**2) / 2)
        cohens_d_monthly = (oh_stats['monthly_mean'] - repo_stats['monthly_mean']) / pooled_std_monthly if pooled_std_monthly > 0 else 0
        
        print(f"{repo}:")
        print(f"  Daily: {repo_stats['daily_mean']:.2f} PRs/day, {repo_stats['commits_per_day']:.2f} Commits/day, B={repo_stats['daily_burstiness']:.4f}, d={cohens_d_daily:.4f}")
        print(f"  Monthly: {repo_stats['monthly_mean']:.2f} PRs/month, {repo_stats['commits_per_month']:.2f} Commits/month, B={repo_stats['monthly_burstiness']:.4f}, d={cohens_d_monthly:.4f}")
        print(f"  Avg Commits/PR: {avg_commits_pr:.2f}")
        
        results.append({
            'dataset': 'AIDev-Time',
            'repo': repo,
            'daily_mean': repo_stats['daily_mean'],
            'daily_burstiness': repo_stats['daily_burstiness'],
            'cohens_d_daily': cohens_d_daily,
            'p_value_daily': p_daily,
            'monthly_mean': repo_stats['monthly_mean'],
            'monthly_burstiness': repo_stats['monthly_burstiness'],
            'cohens_d_monthly': cohens_d_monthly,
            'p_value_monthly': p_monthly,
            'total_prs': len(repo_prs),
            'avg_commits_pr': avg_commits_pr,
            'commits_per_day': repo_stats['commits_per_day'],
            'commits_per_month': repo_stats['commits_per_month']
        })
    
    return pd.DataFrame(results), oh_stats

def main():
    print("="*80)
    print("RQ2: TEMPORAL ACTIVITY PATTERNS (DAILY + MONTHLY + BURSTINESS + COMMITS)")
    print("="*80)
    
    # Load data
    oh_prs, aidev_full, aidev_time, oh_commits, commit_lookup = load_data()
    
    # Calculate OpenHands avg commits per PR
    oh_avg_commits = oh_commits['avg_commits_per_pr'].mean()
    
    # Compare repositories
    results_df, oh_stats = compare_repositories(oh_prs, aidev_full, aidev_time, oh_commits, commit_lookup)
    
    # Save results
    results_df.to_csv('analysis_outputs/rq2_temporal_patterns_combined.csv', index=False)
    print(f"\n✓ Saved: analysis_outputs/rq2_temporal_patterns_combined.csv")
    
    print("\n" + "="*80)
    print("RQ2 COMBINED ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
