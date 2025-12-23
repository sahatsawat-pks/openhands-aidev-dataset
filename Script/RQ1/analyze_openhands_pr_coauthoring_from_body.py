import pandas as pd
import re

def classify_contributor(name):
    """Classify contributor as human, agent, or bot"""
    if pd.isna(name):
        return 'unknown'
    
    name_lower = str(name).lower()
    
    # Bot keywords
    bot_keywords = ['dependabot', 'github-actions', 'renovate', '[bot]', 'bot]', 
                    'web-flow', 'codecov', 'netlify']
    
    # Agent keywords  
    agent_keywords = ['openhands', 'copilot', 'codex', 'gpt', 'claude', 'cursor', 
                      'devin', 'aider', 'cody']
    
    # Check for bots first
    for keyword in bot_keywords:
        if keyword in name_lower:
            return 'bot'
    
    # Check for agents
    for keyword in agent_keywords:
        if keyword in name_lower:
            return 'agent'
    
    return 'human'

def classify_pr_pattern(pr_user, pr_body):
    """Classify PR co-authoring pattern based on user and body"""
    contributors = set()
    
    # Main PR author
    main_type = classify_contributor(pr_user)
    contributors.add(main_type)
    
    # Check PR body for co-authors
    if pd.notna(pr_body):
        body_str = str(pr_body)
        
        # Look for Co-authored-by: lines
        coauthor_pattern = r'Co-[Aa]uthored-[Bb]y:\s*([^<\n]+?)(?:<[^>]+>)?(?:\n|$)'
        coauthors = re.findall(coauthor_pattern, body_str)
        
        for coauthor in coauthors:
            coauthor_type = classify_contributor(coauthor.strip())
            contributors.add(coauthor_type)
        
        # Also check for OpenHands mentions
        body_lower = body_str.lower()
        if 'openhands' in body_lower or 'openhand' in body_lower:
            contributors.add('agent')
    
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

print("Loading OpenHands PR data...")
df = pd.read_csv('analysis_outputs/openhands_prs_2024-03-13_to_2025-06-11.csv')
print(f"Total OpenHands PRs: {len(df)}")

print("\n" + "="*80)
print("Analyzing co-authoring patterns...")
print("="*80)

# Classify each PR
df['coauthoring_pattern'] = df.apply(
    lambda row: classify_pr_pattern(row['user'], row.get('body', '')),
    axis=1
)

# Count patterns
pattern_counts = df['coauthoring_pattern'].value_counts()
print("\nCo-authoring pattern distribution:")
print(pattern_counts)

# Calculate percentages
total_prs = len(df)
print(f"\nCo-authoring pattern percentages (n={total_prs}):")
for pattern, count in pattern_counts.items():
    pct = (count / total_prs) * 100
    print(f"  {pattern}: {count} ({pct:.2f}%)")

# Save detailed results
output_file = 'analysis_outputs/openhands_pr_coauthoring_patterns.csv'
df[['number', 'title', 'user', 'state', 'created_at', 'coauthoring_pattern']].to_csv(
    output_file, index=False
)
print(f"\n✓ Detailed results saved to: {output_file}")

# Save summary
summary_df = pd.DataFrame({
    'Pattern': pattern_counts.index,
    'Count': pattern_counts.values,
    'Percentage': (pattern_counts.values / total_prs * 100).round(2)
})
summary_file = 'analysis_outputs/openhands_pr_coauthoring_summary_body_based.csv'
summary_df.to_csv(summary_file, index=False)
print(f"✓ Summary saved to: {summary_file}")

print("\n" + "="*80)
print("Sample PRs by pattern:")
print("="*80)
for pattern in pattern_counts.head(4).index:
    print(f"\n{pattern}:")
    sample = df[df['coauthoring_pattern'] == pattern].head(2)
    for _, row in sample.iterrows():
        print(f"  - PR #{row['number']}: {row['title'][:60]}...")
        print(f"    User: {row['user']}")
