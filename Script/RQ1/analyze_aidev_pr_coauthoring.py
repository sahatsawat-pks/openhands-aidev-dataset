import pandas as pd
import re
from collections import Counter

def classify_contributor_type(name, user, body):
    """Classify a contributor as human, agent, or bot"""
    if not name and not user:
        return 'unknown'
    
    # Combine all text fields for checking
    all_text = f"{name} {user} {body}".lower() if body else f"{name} {user}".lower()
    
    # Bot keywords
    bot_keywords = ['dependabot', 'github-actions', 'renovate', 'web-flow', 
                    'github-action', 'codecov', 'netlify', 'vercel-bot']
    
    # Agent keywords - these indicate AI agent involvement
    agent_keywords = ['copilot', 'codex', 'gpt', 'claude', 'cursor', 'devin', 
                      'openhands', 'aider', 'continue', 'cody', 'ghostwriter',
                      'tabnine', 'amazon q', 'gemini code', 'starcoder',
                      'claude code', 'ai-generated', 'co-authored-by: claude',
                      'co-authored-by: github copilot', 'generated with']
    
    # Check for bots first (most specific)
    for keyword in bot_keywords:
        if keyword in all_text:
            return 'bot'
    
    # Check for agents
    for keyword in agent_keywords:
        if keyword in all_text:
            return 'agent'
    
    # Default to human
    return 'human'

def extract_coauthors_from_body(body):
    """Extract co-authors from PR body"""
    if not body or pd.isna(body):
        return []
    
    # Look for Co-authored-by: lines
    coauthor_pattern = r'Co-[Aa]uthored-[Bb]y:\s*([^<\n]+?)(?:<[^>]+>)?(?:\n|$)'
    matches = re.findall(coauthor_pattern, str(body))
    
    return [m.strip() for m in matches if m.strip()]

def classify_pr_coauthoring_pattern(agent, user, body):
    """Classify a PR's co-authoring pattern"""
    contributors = set()
    
    # Check the main user
    main_type = classify_contributor_type(user, user, body)
    contributors.add(main_type)
    
    # Check if PR has agent tag
    if pd.notna(agent) and agent:
        contributors.add('agent')
    
    # Check co-authors in body
    coauthors = extract_coauthors_from_body(body)
    for coauthor in coauthors:
        coauthor_type = classify_contributor_type(coauthor, coauthor, body)
        contributors.add(coauthor_type)
    
    # Remove unknown if other types exist
    if 'unknown' in contributors and len(contributors) > 1:
        contributors.remove('unknown')
    
    # Classify the pattern
    has_human = 'human' in contributors
    has_agent = 'agent' in contributors
    has_bot = 'bot' in contributors
    
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

print("Loading AIDev dataset from parquet...")
df = pd.read_parquet('all_pull_request.parquet')

print(f"Total PRs: {len(df):,}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df[['agent', 'user', 'body']].head(3))

print("\n" + "="*80)
print("Analyzing co-authoring patterns...")
print("="*80)

# Classify each PR
df['coauthoring_pattern'] = df.apply(
    lambda row: classify_pr_coauthoring_pattern(row['agent'], row['user'], row['body']),
    axis=1
)

# Count patterns
pattern_counts = df['coauthoring_pattern'].value_counts()
print("\nCo-authoring pattern distribution:")
print(pattern_counts)

# Calculate percentages
total_prs = len(df)
print(f"\nCo-authoring pattern percentages:")
for pattern, count in pattern_counts.items():
    pct = (count / total_prs) * 100
    print(f"  {pattern}: {count:,} ({pct:.2f}%)")

# Save detailed results
output_file = 'analysis_outputs/aidev_pr_coauthoring_patterns.csv'
df[['id', 'number', 'title', 'agent', 'user', 'repo_url', 'coauthoring_pattern']].to_csv(
    output_file, index=False
)
print(f"\n✓ Detailed results saved to: {output_file}")

# Save summary
summary_df = pd.DataFrame({
    'Pattern': pattern_counts.index,
    'Count': pattern_counts.values,
    'Percentage': (pattern_counts.values / total_prs * 100).round(2)
})
summary_file = 'analysis_outputs/aidev_pr_coauthoring_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"✓ Summary saved to: {summary_file}")

# Check agent distribution
print("\n" + "="*80)
print("Agent distribution in dataset:")
print("="*80)
agent_counts = df['agent'].value_counts()
print(agent_counts.head(20))

print("\n" + "="*80)
print("Sample PRs by pattern:")
print("="*80)
for pattern in pattern_counts.head(5).index:
    print(f"\n{pattern}:")
    sample = df[df['coauthoring_pattern'] == pattern].head(2)
    for _, row in sample.iterrows():
        print(f"  - PR #{row['number']} by {row['user']}, agent={row['agent']}")
        if row['body']:
            body_preview = str(row['body'])[:200].replace('\n', ' ')
            print(f"    Body: {body_preview}...")
