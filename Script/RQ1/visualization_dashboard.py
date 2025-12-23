"""
Visualization Dashboard for RQ1: PR Behavioral Pattern Analysis

Creates comprehensive visualizations for clustering results and pattern analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class VisualizationDashboard:
    """Create visualizations for clustering analysis."""
    
    def __init__(self, output_dir='visualizations'):
        """Initialize visualization dashboard."""
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_2d_projection(
        self,
        projection: np.ndarray,
        labels: np.ndarray,
        method_name: str,
        label_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot 2D projection colored by labels.
        
        Args:
            projection: 2D projection array
            labels: Agent/cluster labels
            method_name: Name of projection method (PCA, t-SNE, UMAP)
            label_names: Names for labels
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique labels and colors
        unique_labels = sorted(set(labels))
        colors = sns.color_palette('husl', len(unique_labels))
        
        # Plot each label
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = label_names[i] if label_names else f'Label {label}'
            
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                c=[colors[i]],
                label=label_name,
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
        
        ax.set_xlabel(f'{method_name} Component 1', fontsize=12)
        ax.set_ylabel(f'{method_name} Component 2', fontsize=12)
        ax.set_title(f'{method_name} Projection Colored by Agent', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    def plot_cluster_comparison(
        self,
        projection: np.ndarray,
        true_labels: np.ndarray,
        cluster_labels: np.ndarray,
        method_name: str,
        agent_names: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot side-by-side comparison of true agents vs predicted clusters.
        
        Args:
            projection: 2D projection
            true_labels: True agent labels
            cluster_labels: Predicted cluster labels
            method_name: Projection method name
            agent_names: Agent name list
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left: True labels (agents)
        unique_agents = sorted(set(true_labels))
        agent_colors = sns.color_palette('husl', len(unique_agents))
        
        for i, agent in enumerate(unique_agents):
            mask = true_labels == agent
            axes[0].scatter(
                projection[mask, 0],
                projection[mask, 1],
                c=[agent_colors[i]],
                label=agent_names[i] if i < len(agent_names) else f'Agent {agent}',
                alpha=0.6,
                s=20
            )
        
        axes[0].set_xlabel(f'{method_name} Component 1', fontsize=12)
        axes[0].set_ylabel(f'{method_name} Component 2', fontsize=12)
        axes[0].set_title('True Agent Labels', fontsize=14, fontweight='bold')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Right: Predicted clusters
        unique_clusters = sorted(set(cluster_labels))
        cluster_colors = sns.color_palette('Set2', len(unique_clusters))
        
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            label = f'Cluster {cluster}' if cluster != -1 else 'Noise'
            axes[1].scatter(
                projection[mask, 0],
                projection[mask, 1],
                c=[cluster_colors[i]],
                label=label,
                alpha=0.6,
                s=20
            )
        
        axes[1].set_xlabel(f'{method_name} Component 1', fontsize=12)
        axes[1].set_ylabel(f'{method_name} Component 2', fontsize=12)
        axes[1].set_title('Predicted Clusters', fontsize=14, fontweight='bold')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    def plot_dendrogram(
        self,
        features_df: pd.DataFrame,
        method='ward',
        save_path: Optional[str] = None,
        max_samples=1000
    ):
        """
        Plot hierarchical clustering dendrogram.
        
        Args:
            features_df: Feature DataFrame
            method: Linkage method
            save_path: Path to save figure
            max_samples: Max samples for performance
        """
        # Sample if too many
        if len(features_df) > max_samples:
            features_sample = features_df.sample(n=max_samples, random_state=42)
        else:
            features_sample = features_df
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Compute linkage
        linkage_matrix = linkage(features_sample, method=method)
        
        # Plot dendrogram
        dendrogram(
            linkage_matrix,
            ax=ax,
            no_labels=True,
            color_threshold=0
        )
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        ax.set_title(f'Hierarchical Clustering Dendrogram ({method} linkage)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    def plot_elbow_curve(
        self,
        k_range: range,
        inertias: List[float],
        save_path: Optional[str] = None
    ):
        """
        Plot elbow curve for K-means.
        
        Args:
            k_range: Range of K values
            inertias: Inertia values
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
        ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        loading_df: pd.DataFrame,
        top_n=20,
        save_path: Optional[str] = None
    ):
        """
        Plot PCA feature importance (loadings).
        
        Args:
            loading_df: DataFrame with feature loadings
            top_n: Number of top features to show
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top features for PC1
        top_pc1 = loading_df.nlargest(top_n, 'PC1')['PC1']
        axes[0].barh(range(len(top_pc1)), top_pc1.values, color='steelblue')
        axes[0].set_yticks(range(len(top_pc1)))
        axes[0].set_yticklabels(top_pc1.index, fontsize=9)
        axes[0].set_xlabel('Loading Value', fontsize=12)
        axes[0].set_title(f'Top {top_n} Features for PC1', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].invert_yaxis()
        
        # Top features for PC2
        top_pc2 = loading_df.nlargest(top_n, 'PC2')['PC2']
        axes[1].barh(range(len(top_pc2)), top_pc2.values, color='coral')
        axes[1].set_yticks(range(len(top_pc2)))
        axes[1].set_yticklabels(top_pc2.index, fontsize=9)
        axes[1].set_xlabel('Loading Value', fontsize=12)
        axes[1].set_title(f'Top {top_n} Features for PC2', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    def plot_feature_distributions(
        self,
        features_df: pd.DataFrame,
        agent_labels: np.ndarray,
        agent_names: List[str],
        selected_features: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot feature distributions per agent.
        
        Args:
            features_df: Feature DataFrame
            agent_labels: Agent labels
            agent_names: Agent names
            selected_features: Features to plot
            save_path: Path to save figure
        """
        n_features = len(selected_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        df_plot = features_df[selected_features].copy()
        df_plot['agent'] = agent_labels
        
        for i, feature in enumerate(selected_features):
            ax = axes[i]
            
            # Violin plot
            sns.violinplot(
                data=df_plot,
                x='agent',
                y=feature,
                ax=ax,
                palette='husl'
            )
            
            ax.set_xlabel('Agent', fontsize=10)
            ax.set_ylabel(feature, fontsize=10)
            ax.set_title(f'{feature} Distribution', fontsize=11, fontweight='bold')
            ax.set_xticklabels([agent_names[int(l.get_text())] for l in ax.get_xticklabels()], rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused axes
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(
        self,
        true_labels: np.ndarray,
        cluster_labels: np.ndarray,
        agent_names: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix of clusters vs agents.
        
        Args:
            true_labels: True agent labels
            cluster_labels: Predicted cluster labels
            agent_names: Agent names
            save_path: Path to save figure
        """
        from sklearn.metrics import confusion_matrix
        
        # Filter noise
        mask = cluster_labels != -1
        true_labels_filtered = true_labels[mask]
        cluster_labels_filtered = cluster_labels[mask]
        
        cm = confusion_matrix(true_labels_filtered, cluster_labels_filtered)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            xticklabels=[f'Cluster {i}' for i in range(cm.shape[1])],
            yticklabels=agent_names,
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_xlabel('Predicted Cluster', fontsize=12)
        ax.set_ylabel('True Agent', fontsize=12)
        ax.set_title('Confusion Matrix: Agents vs Clusters', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    def plot_cluster_composition(
        self,
        cluster_labels: np.ndarray,
        agent_labels: np.ndarray,
        agent_names: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot cluster composition (which agents are in each cluster).
        
        Args:
            cluster_labels: Cluster labels
            agent_labels: Agent labels
            agent_names: Agent names
            save_path: Path to save figure
        """
        # Create composition DataFrame
        df = pd.DataFrame({
            'cluster': cluster_labels,
            'agent': agent_labels
        })
        
        # Filter noise
        df = df[df['cluster'] != -1]
        
        # Calculate composition
        composition = df.groupby(['cluster', 'agent']).size().unstack(fill_value=0)
        composition_pct = composition.div(composition.sum(axis=1), axis=0) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute counts
        composition.plot(
            kind='bar',
            stacked=True,
            ax=axes[0],
            color=sns.color_palette('husl', len(agent_names)),
            width=0.7
        )
        axes[0].set_xlabel('Cluster', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Cluster Composition (Absolute)', fontsize=13, fontweight='bold')
        axes[0].legend(agent_names, title='Agent', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Percentage
        composition_pct.plot(
            kind='bar',
            stacked=True,
            ax=axes[1],
            color=sns.color_palette('husl', len(agent_names)),
            width=0.7
        )
        axes[1].set_xlabel('Cluster', fontsize=12)
        axes[1].set_ylabel('Percentage (%)', fontsize=12)
        axes[1].set_title('Cluster Composition (Percentage)', fontsize=13, fontweight='bold')
        axes[1].legend(agent_names, title='Agent', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_ylim([0, 100])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig


def test_visualization():
    """Test visualization functions."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    
    # Create 2D projection
    projection = np.random.randn(n_samples, 2)
    labels = np.random.choice([0, 1, 2, 3, 4], size=n_samples)
    agent_names = ['OpenAI Codex', 'Devin', 'GitHub Copilot', 'Cursor', 'Claude Code']
    
    viz = VisualizationDashboard(output_dir='test_viz')
    
    # Test 2D projection plot
    viz.plot_2d_projection(
        projection,
        labels,
        'PCA',
        label_names=agent_names,
        save_path='test_viz/test_pca.png'
    )
    
    print("Visualization test complete!")


if __name__ == '__main__':
    test_visualization()
