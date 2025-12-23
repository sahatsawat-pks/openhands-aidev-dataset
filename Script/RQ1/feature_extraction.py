"""
Feature Extraction Pipeline for RQ1: PR Behavioral Pattern Analysis

Extracts multidimensional behavioral features from PR data to identify
distinguishable patterns across different AI coding agents.
"""

import re
import emoji
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class PRFeatureExtractor:
    """Extract behavioral features from Pull Request data."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = []
        
    def extract_text_structure_features(self, text: str) -> Dict[str, float]:
        """
        Extract structural features from PR body text.
        
        Args:
            text: PR body text (markdown formatted)
            
        Returns:
            Dictionary of text structure features
        """
        if pd.isna(text) or not isinstance(text, str):
            text = ""
            
        features = {}
        
        # Basic length metrics
        features['body_length'] = len(text)
        features['body_word_count'] = len(text.split())
        features['body_line_count'] = len(text.split('\n'))
        
        # Markdown structure
        features['heading_count'] = len(re.findall(r'^#{1,6}\s+', text, re.MULTILINE))
        features['bullet_list_count'] = len(re.findall(r'^\s*[-*+]\s+', text, re.MULTILINE))
        features['numbered_list_count'] = len(re.findall(r'^\s*\d+\.\s+', text, re.MULTILINE))
        features['checkbox_count'] = len(re.findall(r'- \[[ xX]\]', text))
        
        # Links and references
        features['link_count'] = len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text))
        features['image_count'] = len(re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', text))
        features['mention_count'] = len(re.findall(r'@[\w-]+', text))
        features['issue_reference_count'] = len(re.findall(r'#\d+', text))
        
        # Formatting
        features['bold_count'] = len(re.findall(r'\*\*[^*]+\*\*|__[^_]+__', text))
        features['italic_count'] = len(re.findall(r'\*[^*]+\*|_[^_]+_', text))
        features['blockquote_count'] = len(re.findall(r'^>\s+', text, re.MULTILINE))
        
        # Paragraph structure
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        features['paragraph_count'] = len(paragraphs)
        features['avg_paragraph_length'] = np.mean([len(p) for p in paragraphs]) if paragraphs else 0
        
        return features
    
    def extract_code_block_features(self, text: str) -> Dict[str, float]:
        """
        Extract code block related features.
        
        Args:
            text: PR body text
            
        Returns:
            Dictionary of code block features
        """
        if pd.isna(text) or not isinstance(text, str):
            text = ""
            
        features = {}
        
        # Code fences (```language ... ```)
        code_blocks = re.findall(r'```(\w*)\n(.*?)```', text, re.DOTALL)
        features['code_block_count'] = len(code_blocks)
        
        # Code block sizes
        if code_blocks:
            block_sizes = [len(block[1]) for block in code_blocks]
            features['avg_code_block_size'] = np.mean(block_sizes)
            features['max_code_block_size'] = np.max(block_sizes)
            features['total_code_chars'] = sum(block_sizes)
        else:
            features['avg_code_block_size'] = 0
            features['max_code_block_size'] = 0
            features['total_code_chars'] = 0
        
        # Language diversity
        languages = [block[0].lower() for block in code_blocks if block[0]]
        features['unique_languages_count'] = len(set(languages))
        
        # Common languages (binary features)
        common_langs = ['python', 'javascript', 'typescript', 'java', 'bash', 'sql', 'json', 'yaml']
        for lang in common_langs:
            features[f'has_{lang}_code'] = 1.0 if lang in languages else 0.0
        
        # Inline code (single backticks)
        inline_code = re.findall(r'`[^`]+`', text)
        features['inline_code_count'] = len(inline_code)
        
        # Code to text ratio
        total_chars = len(text)
        features['code_to_text_ratio'] = features['total_code_chars'] / total_chars if total_chars > 0 else 0
        
        return features
    
    def extract_emoji_features(self, text: str) -> Dict[str, float]:
        """
        Extract emoji usage features.
        
        Args:
            text: PR body text
            
        Returns:
            Dictionary of emoji features
        """
        if pd.isna(text) or not isinstance(text, str):
            text = ""
            
        features = {}
        
        # Extract all emojis
        emojis_found = emoji.emoji_list(text)
        features['emoji_count'] = len(emojis_found)
        features['unique_emoji_count'] = len(set([e['emoji'] for e in emojis_found]))
        
        # Emoji density
        features['emoji_density'] = len(emojis_found) / len(text) if len(text) > 0 else 0
        
        # Common emoji categories (check for common patterns)
        emoji_chars = [e['emoji'] for e in emojis_found]
        features['has_checkmark_emoji'] = 1.0 if any(e in ['✅', '✓', '☑'] for e in emoji_chars) else 0.0
        features['has_rocket_emoji'] = 1.0 if '🚀' in emoji_chars else 0.0
        features['has_bug_emoji'] = 1.0 if '🐛' in emoji_chars else 0.0
        features['has_sparkle_emoji'] = 1.0 if '✨' in emoji_chars else 0.0
        
        return features
    
    def extract_title_features(self, title: str) -> Dict[str, float]:
        """
        Extract features from PR title.
        
        Args:
            title: PR title
            
        Returns:
            Dictionary of title features
        """
        if pd.isna(title) or not isinstance(title, str):
            title = ""
            
        features = {}
        
        features['title_length'] = len(title)
        features['title_word_count'] = len(title.split())
        
        # Conventional commit prefixes
        prefixes = ['feat:', 'fix:', 'chore:', 'docs:', 'style:', 'refactor:', 'test:', 'perf:']
        features['has_conventional_prefix'] = 1.0 if any(title.lower().startswith(p) for p in prefixes) else 0.0
        
        # Emoji in title
        title_emojis = emoji.emoji_list(title)
        features['title_has_emoji'] = 1.0 if len(title_emojis) > 0 else 0.0
        
        # All caps words (excluding common acronyms)
        caps_words = re.findall(r'\b[A-Z]{2,}\b', title)
        features['title_caps_word_count'] = len([w for w in caps_words if len(w) > 3])
        
        return features
    
    def extract_metadata_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Extract metadata features from PR.
        
        Args:
            row: DataFrame row with PR data
            
        Returns:
            Dictionary of metadata features
        """
        features = {}
        
        # Merge status
        features['is_merged'] = 1.0 if pd.notna(row.get('merged_at')) else 0.0
        features['is_closed'] = 1.0 if row.get('state') == 'closed' else 0.0
        
        # PR lifetime
        if pd.notna(row.get('created_at')) and pd.notna(row.get('closed_at')):
            created = pd.to_datetime(row['created_at'])
            closed = pd.to_datetime(row['closed_at'])
            lifetime_hours = (closed - created).total_seconds() / 3600
            features['pr_lifetime_hours'] = lifetime_hours
        else:
            features['pr_lifetime_hours'] = 0
        
        # Time of creation (cyclical features)
        if pd.notna(row.get('created_at')):
            created = pd.to_datetime(row['created_at'])
            features['created_hour'] = created.hour
            features['created_day_of_week'] = created.dayofweek
            features['created_month'] = created.month
        else:
            features['created_hour'] = 0
            features['created_day_of_week'] = 0
            features['created_month'] = 0
        
        return features
    
    def extract_all_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Extract all features from a PR row.
        
        Args:
            row: DataFrame row with PR data (must have 'title', 'body' columns)
            
        Returns:
            Dictionary of all extracted features
        """
        all_features = {}
        
        # Extract from different components
        title = row.get('title', '')
        body = row.get('body', '')
        
        all_features.update(self.extract_title_features(title))
        all_features.update(self.extract_text_structure_features(body))
        all_features.update(self.extract_code_block_features(body))
        all_features.update(self.extract_emoji_features(body))
        all_features.update(self.extract_metadata_features(row))
        
        return all_features
    
    def extract_features_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from a batch of PRs.
        
        Args:
            df: DataFrame with PR data
            
        Returns:
            DataFrame with extracted features
        """
        print(f"Extracting features from {len(df)} PRs...")
        
        features_list = []
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"  Processed {idx}/{len(df)} PRs...")
            features = self.extract_all_features(row)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list, index=df.index)
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        print(f"✓ Extracted {len(self.feature_names)} features")
        
        return features_df
    
    def normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using StandardScaler.
        
        Args:
            features_df: DataFrame with raw features
            
        Returns:
            DataFrame with normalized features
        """
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        normalized = scaler.fit_transform(features_df)
        
        return pd.DataFrame(
            normalized,
            columns=features_df.columns,
            index=features_df.index
        )


def test_feature_extraction():
    """Test feature extraction on sample data."""
    sample_pr = {
        'title': '🚀 feat: Add new authentication system',
        'body': '''## Summary

This PR adds a new authentication system with JWT support.

### Changes
- Added JWT token generation
- Implemented user login endpoint
- Added password hashing

```python
def authenticate(username, password):
    # Authentication logic
    return True
```

### Testing
- [x] Unit tests added
- [x] Integration tests passed

Fixes #123
@reviewer please review!
''',
        'state': 'closed',
        'created_at': '2024-01-01T10:00:00Z',
        'closed_at': '2024-01-01T15:00:00Z',
        'merged_at': '2024-01-01T15:00:00Z'
    }
    
    extractor = PRFeatureExtractor()
    features = extractor.extract_all_features(pd.Series(sample_pr))
    
    print("Sample Feature Extraction:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print(f"\nTotal features extracted: {len(features)}")


if __name__ == '__main__':
    test_feature_extraction()
