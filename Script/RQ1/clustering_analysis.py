"""
Clustering Analysis for RQ1: Identifying PR Behavioral Patterns

Applies unsupervised learning algorithms (K-means, Hierarchical, DBSCAN)
and dimensionality reduction (PCA, t-SNE, UMAP) to identify patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ClusteringAnalyzer:
    """Perform clustering analysis on PR feature data."""
    
    def __init__(self, random_state=42):
        """Initialize clustering analyzer."""
        self.random_state = random_state
        self.pca_model = None
        self.tsne_model = None
        self.umap_model = None
        self.kmeans_model = None
        
    def apply_pca(self, features_df: pd.DataFrame, n_components=2) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA dimensionality reduction.
        
        Args:
            features_df: Normalized feature DataFrame
            n_components: Number of components to keep
            
        Returns:
            Tuple of (transformed data, fitted PCA model)
        """
        print(f"Applying PCA with {n_components} components...")
        
        pca = PCA(n_components=n_components, random_state=self.random_state)
        transformed = pca.fit_transform(features_df)
        
        variance_explained = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_explained)
        
        print(f"  Variance explained by PC1: {variance_explained[0]:.3f}")
        if n_components >= 2:
            print(f"  Variance explained by PC2: {variance_explained[1]:.3f}")
            print(f"  Cumulative variance (PC1+PC2): {cumulative_variance[1]:.3f}")
        
        self.pca_model = pca
        return transformed, pca
    
    def apply_tsne(self, features_df: pd.DataFrame, n_components=2, perplexity=30) -> np.ndarray:
        """
        Apply t-SNE dimensionality reduction.
        
        Args:
            features_df: Normalized feature DataFrame
            n_components: Number of dimensions
            perplexity: t-SNE perplexity parameter
            
        Returns:
            Transformed data array
        """
        print(f"Applying t-SNE with perplexity={perplexity}...")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=self.random_state,
            max_iter=1000
        )
        
        transformed = tsne.fit_transform(features_df)
        
        print(f"  ✓ t-SNE transformation complete")
        
        self.tsne_model = tsne
        return transformed
    
    def apply_umap(self, features_df: pd.DataFrame, n_components=2, n_neighbors=15) -> np.ndarray:
        """
        Apply UMAP dimensionality reduction.
        
        Args:
            features_df: Normalized feature DataFrame
            n_components: Number of dimensions
            n_neighbors: UMAP n_neighbors parameter
            
        Returns:
            Transformed data array
        """
        try:
            import umap
            
            print(f"Applying UMAP with n_neighbors={n_neighbors}...")
            
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=self.random_state
            )
            
            transformed = reducer.fit_transform(features_df)
            
            print(f"  ✓ UMAP transformation complete")
            
            self.umap_model = reducer
            return transformed
            
        except ImportError:
            print("  ⚠ UMAP not installed. Install with: pip install umap-learn")
            return None
    
    def kmeans_clustering(self, features_df: pd.DataFrame, n_clusters=5) -> Tuple[np.ndarray, KMeans]:
        """
        Apply K-means clustering.
        
        Args:
            features_df: Normalized feature DataFrame
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (cluster labels, fitted model)
        """
        print(f"Applying K-means clustering with k={n_clusters}...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        labels = kmeans.fit_predict(features_df)
        
        # Calculate metrics
        silhouette = silhouette_score(features_df, labels)
        davies_bouldin = davies_bouldin_score(features_df, labels)
        
        print(f"  Silhouette Score: {silhouette:.3f}")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.3f}")
        
        self.kmeans_model = kmeans
        return labels, kmeans
    
    def hierarchical_clustering(self, features_df: pd.DataFrame, n_clusters=5) -> Tuple[np.ndarray, AgglomerativeClustering]:
        """
        Apply hierarchical clustering.
        
        Args:
            features_df: Normalized feature DataFrame
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (cluster labels, fitted model)
        """
        print(f"Applying Hierarchical clustering with {n_clusters} clusters...")
        
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        
        labels = hierarchical.fit_predict(features_df)
        
        # Calculate metrics
        silhouette = silhouette_score(features_df, labels)
        
        print(f"  Silhouette Score: {silhouette:.3f}")
        
        return labels, hierarchical
    
    def dbscan_clustering(self, features_df: pd.DataFrame, eps=0.5, min_samples=5) -> Tuple[np.ndarray, DBSCAN]:
        """
        Apply DBSCAN clustering.
        
        Args:
            features_df: Normalized feature DataFrame
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            
        Returns:
            Tuple of (cluster labels, fitted model)
        """
        print(f"Applying DBSCAN with eps={eps}, min_samples={min_samples}...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features_df)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        
        if n_clusters > 1:
            # Filter out noise for silhouette calculation
            mask = labels != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(features_df[mask], labels[mask])
                print(f"  Silhouette Score (excluding noise): {silhouette:.3f}")
        
        return labels, dbscan
    
    def elbow_analysis(self, features_df: pd.DataFrame, max_k=10) -> List[float]:
        """
        Perform elbow analysis to find optimal K.
        
        Args:
            features_df: Normalized feature DataFrame
            max_k: Maximum K to test
            
        Returns:
            List of inertia values
        """
        print(f"Performing elbow analysis for k=2 to k={max_k}...")
        
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(features_df)
            inertias.append(kmeans.inertia_)
            print(f"  k={k}: inertia={kmeans.inertia_:.2f}")
        
        return inertias
    
    def evaluate_clustering(self, features_df: pd.DataFrame, labels: np.ndarray, true_labels: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate clustering quality.
        
        Args:
            features_df: Feature DataFrame
            labels: Predicted cluster labels
            true_labels: True agent labels (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Filter out noise points for DBSCAN
        mask = labels != -1
        valid_features = features_df[mask]
        valid_labels = labels[mask]
        
        if len(set(valid_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(valid_features, valid_labels)
            metrics['davies_bouldin_index'] = davies_bouldin_score(valid_features, valid_labels)
        else:
            metrics['silhouette_score'] = 0.0
            metrics['davies_bouldin_index'] = 0.0
        
        metrics['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
        metrics['n_noise'] = (labels == -1).sum()
        
        # If true labels provided, calculate ARI
        if true_labels is not None:
            valid_true_labels = true_labels[mask]
            metrics['adjusted_rand_index'] = adjusted_rand_score(valid_true_labels, valid_labels)
        
        return metrics
    
    def calculate_cluster_purity(self, cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
        """
        Calculate cluster purity (dominant class accuracy).
        
        Args:
            cluster_labels: Predicted cluster labels
            true_labels: True agent labels
            
        Returns:
            Purity score (0-1)
        """
        # Filter noise
        mask = cluster_labels != -1
        cluster_labels = cluster_labels[mask]
        true_labels = true_labels[mask]
        
        if len(cluster_labels) == 0:
            return 0.0
        
        purity_sum = 0
        for cluster in set(cluster_labels):
            cluster_mask = cluster_labels == cluster
            cluster_true_labels = true_labels[cluster_mask]
            
            # Find most common true label
            if len(cluster_true_labels) > 0:
                most_common = pd.Series(cluster_true_labels).value_counts().max()
                purity_sum += most_common
        
        purity = purity_sum / len(cluster_labels)
        return purity
    
    def get_feature_importance(self, pca_model: PCA, feature_names: List[str], n_components=2) -> pd.DataFrame:
        """
        Get PCA feature importance (loadings).
        
        Args:
            pca_model: Fitted PCA model
            feature_names: List of feature names
            n_components: Number of components to analyze
            
        Returns:
            DataFrame with feature loadings
        """
        loadings = pca_model.components_[:n_components].T
        
        loading_df = pd.DataFrame(
            loadings,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=feature_names
        )
        
        # Add absolute importance
        loading_df['abs_importance'] = loading_df.abs().sum(axis=1)
        loading_df = loading_df.sort_values('abs_importance', ascending=False)
        
        return loading_df


def test_clustering():
    """Test clustering on sample data."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create synthetic data with 3 clusters
    X = np.random.randn(n_samples, n_features)
    
    analyzer = ClusteringAnalyzer()
    
    # Test PCA
    pca_result, pca_model = analyzer.apply_pca(X, n_components=2)
    print(f"\nPCA result shape: {pca_result.shape}")
    
    # Test K-means
    labels, kmeans = analyzer.kmeans_clustering(X, n_clusters=3)
    print(f"\nK-means clusters: {len(set(labels))}")
    
    # Test evaluation
    metrics = analyzer.evaluate_clustering(X, labels)
    print(f"\nClustering metrics: {metrics}")


if __name__ == '__main__':
    test_clustering()
