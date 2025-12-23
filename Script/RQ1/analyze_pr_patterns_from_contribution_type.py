import pandas as pd
import re

print("="*80)
print("RQ1: PR-Level Co-authoring Pattern Analysis (Using Contribution Type Data)")
print("="*80)

# ==================== OpenHands Analysis ====================
print("\n1. Loading OpenHands commit data with contribution types...")
commits_df = pd.read_csv('Commits/openhand_commits_with_contribution_type.csv')
print(f"Total commits: {len(commits_df):,}")
print(f"Columns: {list(commits_df.columns)}")

# Check contribution type distribution
print("\nContribution type distribution in commits:")
if 'contribution_type' in commits_df.columns:
    print(commits_df['contribution_type'].value_counts())
else:
    print("⚠️ 'contribution_type' column not found")
    print("Available columns:", commits_df.columns.tolist())

# Extract PR numbers from commit messages
def extract_pr_number(message):
    """Extract PR number from commit message"""
    if pd.isna(message):
        return None
    match = re.search(r'#(\d+)', str(message))
    if match:
        return int(match.group(1))
    return None

print("\n2. Extracting PR numbers from commit messages...")
commits_df['pr_number'] = commits_df['message'].apply(extract_pr_number)

# Filter commits with PR numbers
commits_with_pr = commits_df[commits_df['pr_number'].notna()].copy()
print(f"Commits with PR numbers: {len(commits_with_pr):,}")

# Time filter to match the PR dataset period
commits_with_pr['date'] = pd.to_datetime(commits_with_pr['date'])
start_date = '2024-03-13'
end_date = '2025-06-11'
commits_filtered = commits_with_pr[
    (commits_with_pr['date'] >= start_date) & 
    (commits_with_pr['date'] <= end_date)
].copy()
print(f"Commits in time range ({start_date} to {end_date}): {len(commits_filtered):,}")

# Aggregate contribution types per PR
print("\n3. Aggregating contribution types per PR...")

def aggregate_pr_contribution_types(pr_commits):
    """
    Determine PR-level co-authoring pattern based on commit-level contribution types
    Logic: If a PR has multiple commits with different types, combine them
    """
    types = set(pr_commits['contribution_type'].dropna().unique())
    
    # Map contribution types to standardized categories
    has_human = False
    has_agent = False
    has_bot = False
    
    for contrib_type in types:
        contrib_lower = str(contrib_type).lower()
        
        # Check for each category
        if 'human' in contrib_lower and 'agent' not in contrib_lower and 'bot' not in contrib_lower:
            has_human = True
        if 'agent' in contrib_lower:
            has_agent = True
        if 'bot' in contrib_lower:
            has_bot = True
        
        # Handle combined types
        if 'human' in contrib_lower and 'agent' in contrib_lower:
            has_human = True
            has_agent = True
        if 'human' in contrib_lower and 'bot' in contrib_lower:
            has_human = True
            has_bot = True
        if 'agent' in contrib_lower and 'bot' in contrib_lower:
            has_agent = True
            has_bot = True
    
    # Classify the PR pattern
    if has_human and has_agent and has_bot:
        return 'Human + bot + agent'
    elif has_human and has_agent:
        return 'Human + agent'
    elif has_human and has_bot:
        return 'Human + bot'
    elif has_agent and has_bot:
        return 'Bot + agent'
    elif has_agent:
        return 'Agent only'
    elif has_bot:
        return 'Bot only'
    elif has_human:
        return 'Pure human'
    else:
        return 'Unknown'

# Group by PR and determine pattern
pr_patterns = commits_filtered.groupby('pr_number').apply(
    lambda x: pd.Series({
        'pattern': aggregate_pr_contribution_types(x),
        'commit_count': len(x),
        'contribution_types': '|'.join(x['contribution_type'].dropna().unique())
    })
).reset_index()

print(f"Unique PRs analyzed: {len(pr_patterns):,}")

# Count patterns
pattern_counts = pr_patterns['pattern'].value_counts()
print("\n" + "="*80)
print("OpenHands PR Co-authoring Patterns:")
print("="*80)
print(pattern_counts)

total_prs = len(pr_patterns)
print(f"\nPercentages (n={total_prs:,}):")
for pattern, count in pattern_counts.items():
    pct = (count / total_prs) * 100
    print(f"  {pattern}: {count:,} ({pct:.2f}%)")

# Save OpenHands results
openhands_summary = pd.DataFrame({
    'Pattern': pattern_counts.index,
    'Count': pattern_counts.values,
    'Percentage': (pattern_counts.values / total_prs * 100).round(2)
})
openhands_file = 'analysis_outputs/openhands_pr_patterns_from_contribution_type.csv'
openhands_summary.to_csv(openhands_file, index=False)
print(f"\n✓ OpenHands summary saved to: {openhands_file}")

# Save detailed PR patterns
pr_patterns_file = 'analysis_outputs/openhands_pr_patterns_detailed.csv'
pr_patterns.to_csv(pr_patterns_file, index=False)
print(f"✓ Detailed PR patterns saved to: {pr_patterns_file}")

print("\n" + "="*80)
print("Sample PRs by pattern:")
print("="*80)
for pattern in pattern_counts.head(5).index:
    print(f"\n{pattern} ({pattern_counts[pattern]} PRs):")
    samples = pr_patterns[pr_patterns['pattern'] == pattern].head(3)
    for _, row in samples.iterrows():
        print(f"  PR #{int(row['pr_number'])}: {row['commit_count']} commits, types: {row['contribution_types']}")
