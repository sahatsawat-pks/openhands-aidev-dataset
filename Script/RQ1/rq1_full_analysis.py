"""
RQ1 Full-Scale Analysis: Complete AIDev Dataset Analysis

This script runs the complete analysis on the full AIDev dataset (932K PRs)
with additional feature discrimination and outlier analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from feature_extraction import PRFeatureExtractor
from clustering_analysis import ClusteringAnalyzer
from visualization_dashboard import VisualizationDashboard

# Configuration
SAMPLE_SIZE = 100000  # None = use full dataset, or set to integer for sampling
OUTPUT_DIR = Path('../analysis_outputs/rq1_full_analysis')
VIZ_DIR = OUTPUT_DIR / 'visualizations'
RESULTS_DIR = OUTPUT_DIR / 'results'

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("RQ1 FULL-SCALE ANALYSIS")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*80)

# 1. Load AIDev Dataset
print("\n[Step 1/8] Loading AIDev dataset from HuggingFace...")
all_pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_pull_request.parquet")

print(f"✓ Loaded {len(all_pr_df):,} PRs")
print(f"\nAgent distribution:")
agent_counts = all_pr_df['agent'].value_counts()
print(agent_counts)

# Sample if configured
if SAMPLE_SIZE and len(all_pr_df) > SAMPLE_SIZE:
    print(f"\nSampling {SAMPLE_SIZE:,} PRs (stratified by agent)...")
    sampled_df = all_pr_df.groupby('agent', group_keys=False).apply(
        lambda x: x.sample(min(len(x), SAMPLE_SIZE // 5), random_state=42)
    )
    print(f"✓ Sampled {len(sampled_df):,} PRs")
else:
    sampled_df = all_pr_df.copy()
    print(f"\n✓ Using full dataset: {len(sampled_df):,} PRs")

# Save dataset info
dataset_info = {
    'total_prs': len(sampled_df),
    'agents': sampled_df['agent'].unique().tolist(),
    'agent_counts': sampled_df['agent'].value_counts().to_dict(),
    'sample_size': SAMPLE_SIZE,
    'analysis_date': datetime.now().isoformat()
}

import json
with open(RESULTS_DIR / 'dataset_info.json', 'w') as f:
    json.dump(dataset_info, f, indent=2)

# 2. Feature Extraction
print("\n[Step 2/8] Extracting features...")
extractor = PRFeatureExtractor()
features_df = extractor.extract_features_batch(sampled_df)
normalized_features = extractor.normalize_features(features_df)

print(f"✓ Extracted {features_df.shape[1]} features from {features_df.shape[0]:,} PRs")

# Save features
features_df.to_csv(RESULTS_DIR / 'features_raw.csv', index=False)
normalized_features.to_csv(RESULTS_DIR / 'features_normalized.csv', index=False)

# 3. Feature Discrimination Analysis
print("\n[Step 3/8] Analyzing feature discrimination power...")

# Prepare true labels
agents = sorted(sampled_df['agent'].unique())
agent_mapping = {agent: i for i, agent in enumerate(agents)}
true_labels = sampled_df['agent'].map(agent_mapping).values

# Calculate feature importance via statistical tests
from scipy.stats import f_oneway, kruskal

discrimination_results = []

for feature in features_df.columns:
    # Group feature values by agent
    groups = [features_df[feature][sampled_df['agent'] == agent].values 
              for agent in agents]
    
    # ANOVA F-statistic (parametric)
    try:
        f_stat, p_value_anova = f_oneway(*groups)
    except:
        f_stat, p_value_anova = 0, 1.0
    
    # Kruskal-Wallis H-test (non-parametric)
    try:
        h_stat, p_value_kruskal = kruskal(*groups)
    except:
        h_stat, p_value_kruskal = 0, 1.0
    
    # Effect size (eta-squared)
    ss_between = sum([len(g) * (np.mean(g) - np.mean(features_df[feature]))**2 for g in groups])
    ss_total = np.sum((features_df[feature] - np.mean(features_df[feature]))**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    discrimination_results.append({
        'feature': feature,
        'f_statistic': f_stat,
        'p_value_anova': p_value_anova,
        'h_statistic': h_stat,
        'p_value_kruskal': p_value_kruskal,
        'eta_squared': eta_squared,
        'significant': p_value_anova < 0.001
    })

discrimination_df = pd.DataFrame(discrimination_results)
discrimination_df = discrimination_df.sort_values('eta_squared', ascending=False)

print(f"\n✓ Top 10 Most Discriminative Features (by effect size):")
for idx, row in discrimination_df.head(10).iterrows():
    sig = "***" if row['significant'] else ""
    print(f"  {row['feature']}: η² = {row['eta_squared']:.4f} {sig}")

# Save discrimination results
discrimination_df.to_csv(RESULTS_DIR / 'feature_discrimination.csv', index=False)

# 4. Dimensionality Reduction
print("\n[Step 4/8] Applying dimensionality reduction...")
analyzer = ClusteringAnalyzer(random_state=42)

pca_2d, pca_model = analyzer.apply_pca(normalized_features, n_components=2)
tsne_2d = analyzer.apply_tsne(normalized_features, n_components=2, perplexity=30)

try:
    umap_2d = analyzer.apply_umap(normalized_features, n_components=2, n_neighbors=15)
except:
    print("  ⚠ UMAP not available, skipping")
    umap_2d = None

# 5. Clustering
print("\n[Step 5/8] Applying clustering algorithms...")

kmeans_labels, kmeans_model = analyzer.kmeans_clustering(normalized_features, n_clusters=len(agents))
hierarchical_labels, _ = analyzer.hierarchical_clustering(normalized_features, n_clusters=len(agents))

# Calculate metrics
kmeans_metrics = analyzer.evaluate_clustering(normalized_features, kmeans_labels, true_labels)
kmeans_purity = analyzer.calculate_cluster_purity(kmeans_labels, true_labels)

hierarchical_metrics = analyzer.evaluate_clustering(normalized_features, hierarchical_labels, true_labels)
hierarchical_purity = analyzer.calculate_cluster_purity(hierarchical_labels, true_labels)

print(f"\n✓ K-means: Silhouette = {kmeans_metrics['silhouette_score']:.3f}, Purity = {kmeans_purity:.1%}")
print(f"✓ Hierarchical: Silhouette = {hierarchical_metrics['silhouette_score']:.3f}, Purity = {hierarchical_purity:.1%}")

# 6. Outlier Analysis
print("\n[Step 6/8] Investigating outlier PRs...")

# Identify outliers: PRs where cluster doesn't match true agent
outlier_mask = (kmeans_labels != true_labels)
outlier_prs = sampled_df[outlier_mask].copy()
outlier_prs['true_agent'] = sampled_df.loc[outlier_mask, 'agent']
outlier_prs['predicted_cluster'] = kmeans_labels[outlier_mask]
outlier_prs['true_cluster'] = true_labels[outlier_mask]

print(f"✓ Found {len(outlier_prs):,} outlier PRs ({len(outlier_prs)/len(sampled_df)*100:.1f}%)")

# Analyze outlier characteristics
outlier_features = features_df[outlier_mask]
normal_features = features_df[~outlier_mask]

outlier_stats = pd.DataFrame({
    'feature': features_df.columns,
    'outlier_mean': outlier_features.mean(),
    'normal_mean': normal_features.mean(),
    'difference': outlier_features.mean() - normal_features.mean()
})
outlier_stats['abs_difference'] = outlier_stats['difference'].abs()
outlier_stats = outlier_stats.sort_values('abs_difference', ascending=False)

print(f"\nTop 5 features distinguishing outliers from normal PRs:")
for idx, row in outlier_stats.head(5).iterrows():
    print(f"  {row['feature']}: Δ = {row['difference']:.3f}")

# Save outlier analysis
outlier_prs.to_csv(RESULTS_DIR / 'outlier_prs.csv', index=False)
outlier_stats.to_csv(RESULTS_DIR / 'outlier_feature_analysis.csv', index=False)

# Sample interesting outliers for manual inspection
sample_outliers = outlier_prs.groupby('true_agent').head(3)
sample_outliers[['number', 'title', 'true_agent', 'predicted_cluster']].to_csv(
    RESULTS_DIR / 'sample_outliers_for_inspection.csv', 
    index=False
)

# 7. Visualizations
print("\n[Step 7/8] Generating comprehensive visualizations...")
viz = VisualizationDashboard(output_dir=str(VIZ_DIR))

# PCA by agent
viz.plot_2d_projection(pca_2d, true_labels, 'PCA', label_names=agents,
                       save_path=str(VIZ_DIR / 'pca_by_agent.png'))

# t-SNE by agent
viz.plot_2d_projection(tsne_2d, true_labels, 't-SNE', label_names=agents,
                       save_path=str(VIZ_DIR / 'tsne_by_agent.png'))

# UMAP if available
if umap_2d is not None:
    viz.plot_2d_projection(umap_2d, true_labels, 'UMAP', label_names=agents,
                          save_path=str(VIZ_DIR / 'umap_by_agent.png'))

# Feature importance
loading_df = analyzer.get_feature_importance(pca_model, extractor.feature_names, n_components=2)
viz.plot_feature_importance(loading_df, top_n=15,
                           save_path=str(VIZ_DIR / 'feature_importance_pca.png'))

# Cluster comparison
viz.plot_cluster_comparison(pca_2d, true_labels, kmeans_labels, 'PCA', agents,
                           save_path=str(VIZ_DIR / 'cluster_comparison.png'))

# Confusion matrix
viz.plot_confusion_matrix(true_labels, kmeans_labels, agents,
                         save_path=str(VIZ_DIR / 'confusion_matrix.png'))

# Cluster composition
viz.plot_cluster_composition(kmeans_labels, true_labels, agents,
                            save_path=str(VIZ_DIR / 'cluster_composition.png'))

# Top discriminative features distribution
top_features = discrimination_df.head(9)['feature'].tolist()
viz.plot_feature_distributions(features_df, true_labels, agents, top_features,
                              save_path=str(VIZ_DIR / 'top_features_distribution.png'))

plt.close('all')

print(f"✓ Generated visualizations in {VIZ_DIR}/")

# 8. Final Results Summary
print("\n[Step 8/8] Generating final summary...")

# Decision criteria
best_silhouette = max(kmeans_metrics['silhouette_score'], hierarchical_metrics['silhouette_score'])
best_purity = max(kmeans_purity, hierarchical_purity)

if best_silhouette > 0.25 and best_purity > 0.6:
    rq1_answer = "YES"
    confidence = "HIGH"
elif best_silhouette > 0.15 and best_purity > 0.4:
    rq1_answer = "PARTIAL"
    confidence = "MODERATE"
else:
    rq1_answer = "NO"
    confidence = "LOW"

summary = f"""
{'='*80}
RQ1 ANALYSIS SUMMARY
{'='*80}

Dataset:
  • Total PRs analyzed: {len(sampled_df):,}
  • Agents: {len(agents)}
  • Agent distribution: {dict(sampled_df['agent'].value_counts())}

Feature Extraction:
  • Total features: {features_df.shape[1]}
  • Most discriminative feature: {discrimination_df.iloc[0]['feature']} (η² = {discrimination_df.iloc[0]['eta_squared']:.4f})
  • Significant features (p < 0.001): {discrimination_df['significant'].sum()}

Clustering Performance:
  • K-means Silhouette: {kmeans_metrics['silhouette_score']:.3f}
  • K-means Purity: {kmeans_purity:.1%}
  • K-means ARI: {kmeans_metrics.get('adjusted_rand_index', 0):.3f}
  
  • Hierarchical Silhouette: {hierarchical_metrics['silhouette_score']:.3f}
  • Hierarchical Purity: {hierarchical_purity:.1%}

Dimensionality Reduction:
  • PCA variance explained (PC1+PC2): {pca_model.explained_variance_ratio_[:2].sum():.1%}

Outlier Analysis:
  • Outlier PRs: {len(outlier_prs):,} ({len(outlier_prs)/len(sampled_df)*100:.1f}%)
  • Top outlier-distinguishing feature: {outlier_stats.iloc[0]['feature']}

{'='*80}
ANSWER TO RQ1: {rq1_answer} (Confidence: {confidence})
{'='*80}

Interpretation:
"""

if rq1_answer == "YES":
    summary += """
  Different AI coding agents exhibit statistically significant and visually
  distinguishable PR behavior patterns. Clustering algorithms successfully
  identified agent-specific patterns with good cluster quality.
  
  Key Findings:
  • Clear separation in dimensionality reduction plots
  • High cluster purity indicates consistent agent behaviors
  • Multiple discriminative features identified
"""
elif rq1_answer == "PARTIAL":
    summary += """
  Some AI coding agents show distinguishable patterns, but overlap exists.
  Certain agents may share similar PR behavioral characteristics.
  
  Key Findings:
  • Moderate cluster quality suggests partial separation
  • Some agents more distinctive than others
  • Mixed results across different clustering methods
"""
else:
    summary += """
  AI coding agents do not exhibit strongly distinguishable PR behavior patterns.
  Agent behaviors appear more similar than different across analyzed features.
  
  Key Findings:
  • Low cluster quality and purity scores
  • Significant overlap in feature distributions
  • Limited agent-specific behavioral signals
"""

summary += f"""
{'='*80}
Top 10 Discriminative Features:
"""

for idx, row in discrimination_df.head(10).iterrows():
    summary += f"\n  {idx+1}. {row['feature']:<30} (η² = {row['eta_squared']:.4f})"

summary += f"""

{'='*80}
Output Files:
  • Visualizations: {VIZ_DIR}/
  • Results: {RESULTS_DIR}/
  • Feature discrimination: feature_discrimination.csv
  • Outlier analysis: outlier_prs.csv, outlier_feature_analysis.csv
  
Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

print(summary)

# Save summary
with open(RESULTS_DIR / 'analysis_summary.txt', 'w') as f:
    f.write(summary)

# Save metrics
metrics_df = pd.DataFrame([
    {
        'algorithm': 'K-means',
        'silhouette_score': kmeans_metrics['silhouette_score'],
        'davies_bouldin_index': kmeans_metrics['davies_bouldin_index'],
        'adjusted_rand_index': kmeans_metrics.get('adjusted_rand_index', 0),
        'cluster_purity': kmeans_purity,
        'n_clusters': len(agents)
    },
    {
        'algorithm': 'Hierarchical',
        'silhouette_score': hierarchical_metrics['silhouette_score'],
        'davies_bouldin_index': hierarchical_metrics['davies_bouldin_index'],
        'adjusted_rand_index': hierarchical_metrics.get('adjusted_rand_index', 0),
        'cluster_purity': hierarchical_purity,
        'n_clusters': len(agents)
    }
])
metrics_df.to_csv(RESULTS_DIR / 'clustering_metrics.csv', index=False)

print(f"\n✓ All results saved to: {OUTPUT_DIR}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
