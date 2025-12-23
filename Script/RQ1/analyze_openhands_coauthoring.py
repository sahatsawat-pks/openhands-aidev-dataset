import pandas as pd
import ast
import json
from collections import Counter

print("Loading commit data with co-authors...")
commits_df = pd.read_csv('Commits/openhand_commits_with_openhands_coauthor.csv')

print(f"Total commits: {len(commits_df)}")
print(f"Commits with co-authors: {commits_df['has_openhands_coauthor'].sum()}")

# Load OpenHands PRs with commit counts
print("\nLoading PR data...")
prs_df = pd.read_csv('analysis_outputs/openhands_prs_2024-03-13_to_2025-06-11_with_commits.csv')
print(f"Total PRs: {len(prs_df)}")

# Agent and bot keywords for classification
AGENT_KEYWORDS = ['copilot', 'codex', 'gpt', 'claude', 'cursor', 'devin', 'openhands', 'ai']
BOT_KEYWORDS = ['[bot]', 'dependabot', 'github-actions', 'renovate', 'web-flow']

def classify_contributor(name, email=''):
    """Classify a contributor as human, agent, or bot"""
    name_lower = name.lower() if name else ''
    email_lower = email.lower() if email else ''
    
    # Check if it's OpenHands agent
    if 'openhands' in name_lower or 'openhands' in email_lower:
        return 'agent'
    
    # Check if it's a bot
    for bot in BOT_KEYWORDS:
        if bot in name_lower or bot in email_lower:
            return 'bot'
    
    # Check if it's an agent
    for agent in AGENT_KEYWORDS:
        if agent in name_lower or agent in email_lower:
            return 'agent'
    
    return 'human'

# For OpenHands, we need to extract PR numbers from commit messages or use git log
# Since we don't have direct PR-commit mapping, let's use the commit date approach
# Match commits to PRs by date range

print("\nAnalyzing commit authorship patterns...")

# Parse co-authors field
def parse_coauthors(coauthor_str):
    """Parse the co-authors JSON string"""
    if pd.isna(coauthor_str) or coauthor_str == '' or coauthor_str == '[]':
        return []
    try:
        return json.loads(coauthor_str)
    except:
        return []

commits_df['co_authors_parsed'] = commits_df['co_authors'].apply(parse_coauthors)

# Analyze each commit's authorship pattern
def get_commit_contributors(row):
    """Get all contributors for a commit (author + co-authors)"""
    contributors = []
    
    # Main author
    if pd.notna(row['github_author']):
        contrib_type = classify_contributor(row['author_name'], row['author_email'])
        contributors.append({
            'name': row['github_author'],
            'type': contrib_type
        })
    
    # Co-authors
    for coauthor in row['co_authors_parsed']:
        name = coauthor.get('name', '')
        email = coauthor.get('email', '')
        contrib_type = classify_contributor(name, email)
        contributors.append({
            'name': name,
            'type': contrib_type
        })
    
    return contributors

commits_df['all_contributors'] = commits_df.apply(get_commit_contributors, axis=1)

# Get unique contributor types per commit
def get_unique_types(contributors):
    """Get unique contributor types"""
    types = [c['type'] for c in contributors]
    return list(set(types))

commits_df['unique_types'] = commits_df['all_contributors'].apply(get_unique_types)

# Classify co-authoring pattern
def classify_coauthoring_pattern(unique_types):
    """Classify the co-authoring pattern based on contributor types"""
    types_set = set(unique_types)
    
    if len(types_set) == 0:
        return 'Unknown'
    elif len(types_set) == 1:
        if 'human' in types_set:
            return 'Pure human'
        elif 'agent' in types_set:
            return 'Agent only'
        elif 'bot' in types_set:
            return 'Bot only'
    elif len(types_set) == 2:
        if 'human' in types_set and 'agent' in types_set:
            return 'Human + agent'
        elif 'human' in types_set and 'bot' in types_set:
            return 'Human + bot'
        elif 'bot' in types_set and 'agent' in types_set:
            return 'Bot + agent'
    elif len(types_set) == 3:
        return 'Human + bot + agent'
    
    return 'Mixed'

commits_df['coauthoring_pattern'] = commits_df['unique_types'].apply(classify_coauthoring_pattern)

# Print summary statistics
print("\n=== Co-authoring Pattern Summary (Commit Level) ===")
pattern_counts = commits_df['coauthoring_pattern'].value_counts()
for pattern, count in pattern_counts.items():
    percentage = (count / len(commits_df)) * 100
    print(f"{pattern}: {count} commits ({percentage:.1f}%)")

# Count commits with OpenHands co-author
openhands_coauthor_count = commits_df['has_openhands_coauthor'].sum()
print(f"\nCommits with OpenHands co-author: {openhands_coauthor_count} ({(openhands_coauthor_count/len(commits_df))*100:.1f}%)")

# Save detailed commit analysis
output_file = 'analysis_outputs/openhands_commits_coauthoring_patterns.csv'
commits_df[['sha', 'date', 'github_author', 'message', 'has_openhands_coauthor', 
            'all_contributors', 'unique_types', 'coauthoring_pattern']].to_csv(output_file, index=False)
print(f"\n✓ Saved commit-level co-authoring analysis to: {output_file}")

# Create PR-level analysis by aggregating commits
# Since we don't have direct PR mapping, let's estimate based on the filtered PR data
print("\n=== Estimating PR-level co-authoring patterns ===")
print("Note: This is an estimate based on commit distribution within the time period")

# Filter commits to the same time range as PRs
commits_df['date'] = pd.to_datetime(commits_df['date'])
time_min = pd.to_datetime('2024-03-13', utc=True)
time_max = pd.to_datetime('2025-06-11 23:59:59', utc=True)

filtered_commits = commits_df[(commits_df['date'] >= time_min) & (commits_df['date'] <= time_max)].copy()
print(f"Commits in time range (2024-03-13 to 2025-06-11): {len(filtered_commits)}")

print("\n=== Co-authoring Pattern Summary (Time-Filtered Commits) ===")
pattern_counts_filtered = filtered_commits['coauthoring_pattern'].value_counts()
for pattern, count in pattern_counts_filtered.items():
    percentage = (count / len(filtered_commits)) * 100
    print(f"{pattern}: {count} commits ({percentage:.1f}%)")

# Save time-filtered analysis
output_file_filtered = 'analysis_outputs/openhands_commits_coauthoring_patterns_2024-03-13_to_2025-06-11.csv'
filtered_commits.to_csv(output_file_filtered, index=False)
print(f"\n✓ Saved time-filtered commit co-authoring analysis to: {output_file_filtered}")

# Create summary statistics
summary_df = pattern_counts_filtered.reset_index()
summary_df.columns = ['Pattern', 'Count']
summary_df['Percentage'] = (summary_df['Count'] / summary_df['Count'].sum() * 100).round(2)

summary_file = 'analysis_outputs/openhands_coauthoring_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"✓ Saved summary to: {summary_file}")
