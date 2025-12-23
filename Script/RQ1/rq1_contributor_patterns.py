"""
RQ1: Analyze contributor patterns in PRs at repository level
Classify each PR based on commit authors:
- Human only
- Bot only (dependabot, etc.)
- Agent only (openhands-agent, etc.)
- Hybrid (combination of multiple types)

Analyzes:
- OpenHands repository
- Top 5 AI-Dev Full repositories
- Top 4 AI-Dev Time repositories
"""

import pandas as pd
import requests
import time
import re
import os
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

# Configuration
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
REQUESTS_PER_HOUR = 5000
REQUEST_INTERVAL = 3600 / REQUESTS_PER_HOUR

# Per-owner token map (can be provided in .env as JSON in GITHUB_TOKENS or
# as individual vars like GITHUB_TOKEN_ownername)
def load_token_map():
    token_map = {}
    # JSON map in GITHUB_TOKENS
    if os.environ.get('GITHUB_TOKENS'):
        try:
            import json
            token_map = {k.lower(): v for k, v in json.loads(os.environ.get('GITHUB_TOKENS')).items()}
        except Exception:
            token_map = {}

    # Individual env vars GITHUB_TOKEN_<OWNER>
    for k, v in os.environ.items():
        if k.startswith('GITHUB_TOKEN_') and len(k) > len('GITHUB_TOKEN_'):
            owner = k[len('GITHUB_TOKEN_'):].lower()
            token_map[owner] = v

    # default token if provided
    default = os.environ.get('GITHUB_TOKEN')
    return token_map, default

# Global token map & per-token last-request tracking
TOKEN_MAP, DEFAULT_TOKEN = load_token_map()
LAST_REQUEST = defaultdict(lambda: 0.0)

# Bot patterns
BOT_PATTERNS = [
    r'dependabot',
    r'renovate',
    r'\[bot\]',
    r'github-actions',
    r'codecov',
    r'greenkeeper',
    r'snyk-bot'
]

# Agent patterns (AI coding assistants)
AGENT_PATTERNS = [
    r'openhands',
    r'devin',
    r'cursor',
    r'copilot',
    r'codeium',
    r'tabnine',
    r'kite',
    r'ai-agent',
    r'auto-dev',
    r'aider'
]

def is_bot(username, email):
    """Check if commit author is a bot."""
    if not username and not email:
        return False
    
    text = f"{username} {email}".lower()
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in BOT_PATTERNS)

def is_agent(username, email, message):
    """Check if commit author is an AI agent."""
    if not username and not email and not message:
        return False
    
    text = f"{username} {email} {message}".lower()
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in AGENT_PATTERNS)

def fetch_pr_commits(repo_full_name, pr_number):
    """Fetch all commits for a specific PR using per-owner token and rate limiting."""
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/commits"

    owner = repo_full_name.split('/')[0].lower() if '/' in repo_full_name else ''
    token = TOKEN_MAP.get(owner, DEFAULT_TOKEN)
    token_key = token if token else 'anon'

    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token:
        headers['Authorization'] = f'token {token}'

    try:
        # per-token rate limiting
        now = time.time()
        elapsed = now - LAST_REQUEST[token_key]
        wait = max(0, REQUEST_INTERVAL - elapsed)
        if wait > 0:
            time.sleep(wait)

        response = requests.get(url, headers=headers)
        LAST_REQUEST[token_key] = time.time()

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        elif response.status_code == 403:
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            wait_time = max(reset_time - time.time(), 0) + 10
            print(f"  [{owner}] Rate limit hit. Waiting {wait_time:.0f}s...")
            time.sleep(wait_time)
            return fetch_pr_commits(repo_full_name, pr_number)
        else:
            print(f"  [{owner}] Unexpected status {response.status_code} for PR {pr_number}")
            return None
    except Exception as e:
        print(f"  [{owner}] Exception fetching PR {pr_number}: {e}")
        return None


def fetch_all_prs_for_repo(repo_full_name):
    """Fetch all PR metadata for a repository (state=all), paginated.
    Returns list of PR objects (as dicts)."""
    owner = repo_full_name.split('/')[0].lower() if '/' in repo_full_name else ''
    token = TOKEN_MAP.get(owner, DEFAULT_TOKEN)
    token_key = token if token else 'anon'

    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token:
        headers['Authorization'] = f'token {token}'

    prs = []
    page = 1
    per_page = 100
    while True:
        url = f"https://api.github.com/repos/{repo_full_name}/pulls"
        params = {'state': 'all', 'per_page': per_page, 'page': page}

        try:
            now = time.time()
            elapsed = now - LAST_REQUEST[token_key]
            wait = max(0, REQUEST_INTERVAL - elapsed)
            if wait > 0:
                time.sleep(wait)

            response = requests.get(url, headers=headers, params=params)
            LAST_REQUEST[token_key] = time.time()

            if response.status_code == 200:
                page_items = response.json()
                if not page_items:
                    break
                prs.extend(page_items)
                if len(page_items) < per_page:
                    break
                page += 1
                continue
            elif response.status_code == 404:
                print(f"  [{owner}] Repository {repo_full_name} not found (404)")
                break
            elif response.status_code == 403:
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                wait_time = max(reset_time - time.time(), 0) + 10
                print(f"  [{owner}] Rate limit hit when listing PRs. Waiting {wait_time:.0f}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  [{owner}] Unexpected status {response.status_code} listing PRs for {repo_full_name}")
                break
        except Exception as e:
            print(f"  [{owner}] Exception listing PRs for {repo_full_name}: {e}")
            break

    return prs

def classify_contributor(commit_data):
    """Classify a single commit's contributor type."""
    author = commit_data.get('commit', {}).get('author', {})
    author_name = author.get('name', '')
    author_email = author.get('email', '')
    message = commit_data.get('commit', {}).get('message', '')
    
    # Check for bot
    if is_bot(author_name, author_email):
        return 'bot'
    
    # Check for agent
    if is_agent(author_name, author_email, message):
        return 'agent'
    
    # Default to human
    return 'human'

def classify_pr_from_agent_field(agent_field):
    """Classify PR based on agent field from AI-Dev dataset."""
    if pd.isna(agent_field) or agent_field == '':
        return 'human_only'
    
    agent_str = str(agent_field).lower()
    
    # Check for bots
    if any(bot in agent_str for bot in ['dependabot', 'renovate', '[bot]', 'github-actions', 'codecov']):
        return 'bot_only'
    
    # Check for AI agents
    if any(agent in agent_str for agent in ['devin', 'openai', 'codex', 'cursor', 'copilot', 'openhands']):
        return 'agent_only'
    
    return 'human_only'


def compute_burstiness(timestamps):
    """Compute burstiness B = (sigma - mu) / (sigma + mu) from a list/Series of timestamps.
    Timestamps can be strings parseable by pandas.to_datetime. Returns NaN if insufficient data."""
    import numpy as _np
    if timestamps is None or len(timestamps) < 2:
        return _np.nan

    try:
        ts = pd.to_datetime(timestamps).sort_values()
        # convert to seconds
        ts_seconds = ts.astype('int64') / 1e9
        diffs = _np.diff(ts_seconds)
        # drop non-positive diffs
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return _np.nan
        mu = diffs.mean()
        sigma = diffs.std(ddof=0)
        if (mu + sigma) == 0:
            return 0.0
        return (sigma - mu) / (sigma + mu)
    except Exception:
        return _np.nan

def process_aidev_data(df, dataset_name):
    """Process AI-Dev PR data by fetching commits per PR and classifying contributors."""
    print(f"\nProcessing {dataset_name} data by fetching commits...")

    results = []
    total = len(df)
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{total} ({idx/total*100:.1f}%)")

        # Extract repo full name from repo_url
        repo_url = row.get('repo_url', '')
        if isinstance(repo_url, str) and '/' in repo_url:
            repo_full_name = '/'.join(repo_url.split('/')[-2:])
        else:
            repo_full_name = row.get('repo', 'unknown')

        pr_number = int(row.get('number', 0))

        commits = fetch_pr_commits(repo_full_name, pr_number)
        if not commits:
            # fallback: estimate from commit_count if available
            total_commits = int(row.get('commit_count', 1))
            human_commits = 0
            bot_commits = 0
            agent_commits = 0
            pattern = classify_pr_from_agent_field(row.get('agent', ''))
            if pattern == 'agent_only':
                agent_commits = total_commits
            elif pattern == 'bot_only':
                bot_commits = total_commits
            else:
                human_commits = total_commits
        else:
            contributor_types = []
            for c in commits:
                contributor_types.append(classify_contributor(c))

            human_commits = contributor_types.count('human')
            bot_commits = contributor_types.count('bot')
            agent_commits = contributor_types.count('agent')
            total_commits = len(contributor_types)

            types_present = set(contributor_types)
            if len(types_present) == 0:
                pattern = 'unknown'
            elif len(types_present) == 1:
                if 'human' in types_present:
                    pattern = 'human_only'
                elif 'bot' in types_present:
                    pattern = 'bot_only'
                elif 'agent' in types_present:
                    pattern = 'agent_only'
                else:
                    pattern = 'unknown'
            else:
                pattern = 'hybrid'

        results.append({
            'repo': repo_full_name,
            'dataset': dataset_name,
            'pr_number': pr_number,
            'total_commits': total_commits,
            'human_commits': human_commits,
            'bot_commits': bot_commits,
            'agent_commits': agent_commits,
            'pattern': pattern
        })

    print(f"  Processed {len(results)} PRs")
    return pd.DataFrame(results)

def generate_repository_summary(all_results):
    """Generate summary statistics by repository."""
    print("\n" + "="*80)
    print("REPOSITORY-LEVEL CONTRIBUTOR PATTERN SUMMARY")
    print("="*80)
    
    summary_rows = []
    
    for (dataset, repo), group in all_results.groupby(['dataset', 'repo']):
        total_prs = len(group)
        
        # Count patterns
        pattern_counts = group['pattern'].value_counts()
        human_only = pattern_counts.get('human_only', 0)
        bot_only = pattern_counts.get('bot_only', 0)
        agent_only = pattern_counts.get('agent_only', 0)
        hybrid = pattern_counts.get('hybrid', 0)
        unknown = pattern_counts.get('unknown', 0)
        
        # Calculate percentages
        human_only_pct = (human_only / total_prs * 100) if total_prs > 0 else 0
        bot_only_pct = (bot_only / total_prs * 100) if total_prs > 0 else 0
        agent_only_pct = (agent_only / total_prs * 100) if total_prs > 0 else 0
        hybrid_pct = (hybrid / total_prs * 100) if total_prs > 0 else 0
        
        # Total commits by type
        total_commits = group['total_commits'].sum()
        human_commits = group['human_commits'].sum()
        bot_commits = group['bot_commits'].sum()
        agent_commits = group['agent_commits'].sum()
        
        # Average commits per PR
        avg_commits_pr = group['total_commits'].mean()

        # Burstiness based on PR creation times if available
        burst = None
        if 'created_at' in group.columns:
            burst = compute_burstiness(group['created_at'])
        else:
            burst = np.nan
        
        summary_rows.append({
            'Dataset': dataset,
            'Repository': repo.split('/')[-1] if '/' in repo else repo,
            'Total PRs': total_prs,
            'Human Only (%)': human_only_pct,
            'Bot Only (%)': bot_only_pct,
            'Agent Only (%)': agent_only_pct,
            'Hybrid (%)': hybrid_pct,
            'Avg Commits/PR': avg_commits_pr,
            'Burstiness': burst,
            'Total Commits': total_commits,
            'Human Commits': human_commits,
            'Bot Commits': bot_commits,
            'Agent Commits': agent_commits
        })
        
        print(f"\n{repo} ({dataset}):")
        print(f"  Total PRs: {total_prs}")
        print(f"  Human Only: {human_only} ({human_only_pct:.1f}%)")
        print(f"  Bot Only: {bot_only} ({bot_only_pct:.1f}%)")
        print(f"  Agent Only: {agent_only} ({agent_only_pct:.1f}%)")
        print(f"  Hybrid: {hybrid} ({hybrid_pct:.1f}%)")
        print(f"  Avg Commits/PR: {avg_commits_pr:.2f}")
        print(f"  Burstiness: {burst if not pd.isna(burst) else 'N/A'}")
        print(f"  Commits: {human_commits} human, {bot_commits} bot, {agent_commits} agent")
    
    return pd.DataFrame(summary_rows)

def main():
    print("="*80)
    print("RQ1: CONTRIBUTOR PATTERN ANALYSIS - REPOSITORY LEVEL")
    print("="*80)
    
    # Load existing OpenHands contributor pattern data
    print("\nLoading existing OpenHands contributor pattern data...")
    oh_patterns = pd.read_csv('./OpenHands/openhands_contributor_patterns.csv')
    
    # Convert to repository-level format
    oh_results = pd.DataFrame({
        'repo': 'All-Hands-AI/OpenHands',
        'dataset': 'OpenHands',
        'pr_number': oh_patterns['pr_number'],
        'total_commits': oh_patterns['total_commits'],
        'human_commits': oh_patterns['human_commits'],
        'bot_commits': oh_patterns['bot_commits'],
        'agent_commits': oh_patterns['agent_commits'],
        'pattern': oh_patterns['pattern']
    })
    
    print(f"Loaded {len(oh_results)} OpenHands PRs with contributor patterns")
    
    # Load AI-Dev Full repo list (we will fetch all PRs per repo)
    print("\nLoading AI-Dev Full repo list...")
    aidev_full_repos = pd.read_csv('./AIDev-Full/aidev_full_top5_commit_size.csv')
    full_repo_names = list(aidev_full_repos['repo_name'].unique())
    print(f"Found {len(full_repo_names)} AI-Dev Full repos: {full_repo_names}")

    # Load AI-Dev Time repo list
    print("\nLoading AI-Dev Time repo list...")
    aidev_time_repos = pd.read_csv('./AIDev-Time/aidev_time_top4_commit_size.csv')
    time_repo_names = list(aidev_time_repos['repo_name'].unique())
    print(f"Found {len(time_repo_names)} AI-Dev Time repos: {time_repo_names}")
    
    print("\n" + "="*80)
    print("PROCESSING CONTRIBUTOR PATTERNS")
    print("="*80)
    
    # Fetch all PRs for each AI-Dev Full repo and save to CSVs
    print("\nFetching all PRs for AI-Dev Full repos...")
    full_pr_dfs = []
    for repo_full in full_repo_names:
        safe_name = repo_full.replace('/', '__')
        out_path = f'analysis_outputs/aidev_full_{safe_name}_prs.csv'
        if os.path.exists(out_path):
            print(f"  Loading cached PRs for {repo_full} from {out_path}")
            df = pd.read_csv(out_path)
        else:
            print(f"  Fetching PRs for {repo_full}...")
            prs = fetch_all_prs_for_repo(repo_full)
            rows = []
            for p in prs:
                rows.append({
                    'repo': repo_full,
                    'repo_url': f'https://github.com/{repo_full}',
                    'number': p.get('number'),
                    'html_url': p.get('html_url'),
                    'user_login': p.get('user', {}).get('login') if p.get('user') else None,
                    'created_at': p.get('created_at'),
                    'merged_at': p.get('merged_at') if p.get('merged_at') else None
                })
            df = pd.DataFrame(rows)
            df.to_csv(out_path, index=False)
            print(f"  Saved {len(df)} PRs to {out_path}")
        full_pr_dfs.append(df)

    aidev_full_prs = pd.concat(full_pr_dfs, ignore_index=True) if full_pr_dfs else pd.DataFrame()

    # Fetch all PRs for each AI-Dev Time repo and save to CSVs
    print("\nFetching all PRs for AI-Dev Time repos...")
    time_pr_dfs = []
    for repo_full in time_repo_names:
        safe_name = repo_full.replace('/', '__')
        out_path = f'analysis_outputs/aidev_time_{safe_name}_prs.csv'
        if os.path.exists(out_path):
            print(f"  Loading cached PRs for {repo_full} from {out_path}")
            df = pd.read_csv(out_path)
        else:
            print(f"  Fetching PRs for {repo_full}...")
            prs = fetch_all_prs_for_repo(repo_full)
            rows = []
            for p in prs:
                rows.append({
                    'repo': repo_full,
                    'repo_url': f'https://github.com/{repo_full}',
                    'number': p.get('number'),
                    'html_url': p.get('html_url'),
                    'user_login': p.get('user', {}).get('login') if p.get('user') else None,
                    'created_at': p.get('created_at'),
                    'merged_at': p.get('merged_at') if p.get('merged_at') else None
                })
            df = pd.DataFrame(rows)
            df.to_csv(out_path, index=False)
            print(f"  Saved {len(df)} PRs to {out_path}")
        time_pr_dfs.append(df)

    aidev_time_prs = pd.concat(time_pr_dfs, ignore_index=True) if time_pr_dfs else pd.DataFrame()

    # Process AI-Dev datasets (now using fetched PR lists)
    aidev_full_results = process_aidev_data(aidev_full_prs, 'AI-Dev Full')
    aidev_time_results = process_aidev_data(aidev_time_prs, 'AI-Dev Time')
    
    # Combine all results
    all_results = [oh_results, aidev_full_results, aidev_time_results]
    
    # Combine all results
    if len(all_results) == 0:
        print("\nNo results collected!")
        return
    
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    # Save detailed results
    output_csv = 'analysis_outputs/rq1_contributor_patterns_by_repo.csv'
    all_results_df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to: {output_csv}")
    
    # Generate repository summary
    summary_df = generate_repository_summary(all_results_df)
    
    # Save summary
    summary_csv = 'analysis_outputs/rq1_contributor_patterns_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to: {summary_csv}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Total PRs analyzed: {len(all_results_df)}")
    print(f"Repositories analyzed: {all_results_df['repo'].nunique()}")

if __name__ == '__main__':
    main()
