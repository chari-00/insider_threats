"""
Graph-based behavioral analysis using NetworkX
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import logging

from .config import GRAPH_ANOMALY_THRESHOLD, MIN_EDGE_WEIGHT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphAnalyzer:
    """Analyze user-resource interaction graphs for anomalies"""
    
    def __init__(self, df):
        """
        Initialize with authentication logs
        
        Args:
            df (pd.DataFrame): Logs with user_id and resource_id columns
        """
        self.df = df
        self.graph = None
        self.user_metrics = {}
    
    def build_graph(self):
        """
        Build bipartite user-resource interaction graph
        
        Returns:
            nx.Graph: User-resource graph
        """
        logger.info("Building user-resource interaction graph...")
        
        self.graph = nx.Graph()
        
        # Build edges with weights
        edge_weights = defaultdict(lambda: {'weight': 0, 'sensitivity': []})
        
        for _, row in self.df.iterrows():
            user = row['user_id']
            resource = row.get('resource_id', row.get('destination_computer', 'UNKNOWN'))
            sensitivity = row.get('resource_sensitivity', 1)
            
            key = (user, resource)
            edge_weights[key]['weight'] += 1
            edge_weights[key]['sensitivity'].append(sensitivity)
        
        # Add edges to graph
        for (user, resource), data in edge_weights.items():
            if data['weight'] >= MIN_EDGE_WEIGHT:  # Filter weak connections
                avg_sensitivity = np.mean(data['sensitivity'])
                self.graph.add_edge(
                    user, 
                    resource,
                    weight=data['weight'],
                    sensitivity=avg_sensitivity
                )
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def calculate_user_metrics(self):
        """
        Calculate graph-based metrics for each user
        
        Returns:
            dict: User metrics (degree, centrality, sensitivity)
        """
        if self.graph is None:
            raise ValueError("Must call build_graph() first")
        
        logger.info("Calculating graph metrics for users...")
        
        # Get all users (nodes that appear as source in logs)
        users = set(self.df['user_id'].unique())
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness = nx.betweenness_centrality(self.graph)
        
        # Calculate clustering (community structure)
        try:
            clustering = nx.clustering(self.graph)
        except:
            clustering = {node: 0 for node in self.graph.nodes()}
        
        # Per-user metrics
        for user in users:
            if user not in self.graph:
                # User has no connections (shouldn't happen)
                self.user_metrics[user] = {
                    'graph_degree': 0,
                    'graph_centrality': 0,
                    'graph_betweenness': 0,
                    'graph_clustering': 0,
                    'graph_avg_sensitivity': 0,
                    'graph_max_sensitivity': 0,
                    'graph_unique_resources': 0,
                }
                continue
            
            # Get neighbors (resources accessed)
            neighbors = list(self.graph.neighbors(user))
            
            # Sensitivity of accessed resources
            sensitivities = [
                self.graph[user][neighbor].get('sensitivity', 1)
                for neighbor in neighbors
            ]
            
            # Edge weights (access frequency)
            weights = [
                self.graph[user][neighbor]['weight']
                for neighbor in neighbors
            ]
            
            self.user_metrics[user] = {
                'graph_degree': self.graph.degree(user),
                'graph_centrality': degree_centrality.get(user, 0),
                'graph_betweenness': betweenness.get(user, 0),
                'graph_clustering': clustering.get(user, 0),
                'graph_avg_sensitivity': np.mean(sensitivities) if sensitivities else 0,
                'graph_max_sensitivity': np.max(sensitivities) if sensitivities else 0,
                'graph_unique_resources': len(neighbors),
                'graph_avg_access_freq': np.mean(weights) if weights else 0,
                'graph_max_access_freq': np.max(weights) if weights else 0,
            }
        
        logger.info(f"Calculated metrics for {len(self.user_metrics)} users")
        
        return self.user_metrics
    
    def detect_anomalous_connections(self, user_id, baseline_users=None):
        """
        Detect anomalous connections for a specific user
        
        Args:
            user_id (str): User to analyze
            baseline_users (list): List of "normal" users for comparison
            
        Returns:
            dict: Anomaly indicators
        """
        if self.graph is None:
            raise ValueError("Must call build_graph() first")
        
        if user_id not in self.graph:
            return {
                'has_unusual_connections': False,
                'unusual_resources': [],
                'sensitivity_deviation': 0,
            }
        
        user_resources = set(self.graph.neighbors(user_id))
        
        # If baseline users provided, find common resources
        if baseline_users:
            baseline_resources = set()
            for base_user in baseline_users:
                if base_user in self.graph:
                    baseline_resources.update(self.graph.neighbors(base_user))
            
            # Find resources accessed by user but not by baseline
            unusual_resources = user_resources - baseline_resources
        else:
            unusual_resources = []
        
        # Calculate sensitivity deviation
        user_avg_sensitivity = self.user_metrics.get(user_id, {}).get('graph_avg_sensitivity', 0)
        
        all_sensitivities = [
            metrics['graph_avg_sensitivity'] 
            for metrics in self.user_metrics.values()
        ]
        global_avg_sensitivity = np.mean(all_sensitivities) if all_sensitivities else 0
        sensitivity_deviation = user_avg_sensitivity - global_avg_sensitivity
        
        return {
            'has_unusual_connections': len(unusual_resources) > 0,
            'unusual_resources': list(unusual_resources)[:5],  # Top 5
            'num_unusual': len(unusual_resources),
            'sensitivity_deviation': sensitivity_deviation,
        }
    
    def get_user_subgraph(self, user_id, max_neighbors=20):
        """
        Extract subgraph centered on a user (for visualization)
        
        Args:
            user_id (str): User to center on
            max_neighbors (int): Maximum neighbors to include
            
        Returns:
            nx.Graph: Subgraph
        """
        if user_id not in self.graph:
            return nx.Graph()
        
        # Get neighbors
        neighbors = list(self.graph.neighbors(user_id))[:max_neighbors]
        
        # Create subgraph with user + neighbors
        nodes = [user_id] + neighbors
        subgraph = self.graph.subgraph(nodes).copy()
        
        return subgraph
    
    def merge_with_features(self, features_df):
        """
        Merge graph metrics with existing feature DataFrame
        
        Args:
            features_df (pd.DataFrame): User features
            
        Returns:
            pd.DataFrame: Features with graph metrics added
        """
        if not self.user_metrics:
            self.calculate_user_metrics()
        
        # Convert metrics dict to DataFrame
        graph_df = pd.DataFrame.from_dict(self.user_metrics, orient='index')
        graph_df.reset_index(inplace=True)
        graph_df.rename(columns={'index': 'user_id'}, inplace=True)
        
        # Merge
        merged = pd.merge(features_df, graph_df, on='user_id', how='left')
        
        # Fill NaN with 0 for users without graph data
        graph_cols = [col for col in graph_df.columns if col.startswith('graph_')]
        merged[graph_cols] = merged[graph_cols].fillna(0)
        
        logger.info(f"Merged graph features. New shape: {merged.shape}")
        
        return merged


def analyze_graph_features(df, features_df):
    """
    Convenience function to analyze graph and merge with features
    
    Args:
        df (pd.DataFrame): Authentication logs
        features_df (pd.DataFrame): User features
        
    Returns:
        pd.DataFrame: Features with graph metrics
        GraphAnalyzer: Analyzer object (for visualization)
    """
    analyzer = GraphAnalyzer(df)
    analyzer.build_graph()
    analyzer.calculate_user_metrics()
    
    features_with_graph = analyzer.merge_with_features(features_df)
    
    return features_with_graph, analyzer


def build_auth_graph(df):
    # Implement NetworkX graph construction here
    pass


if __name__ == "__main__":
    # Test graph analysis
    from .data_loader import load_all_data
    from .feature_engineering import engineer_features_from_logs
    
    logger.info("Loading data...")
    df = load_all_data(use_lanl=False, use_synthetic=True)
    
    if df is not None:
        logger.info("Engineering features...")
        features_df = engineer_features_from_logs(df)
        
        logger.info("Analyzing graph...")
        features_with_graph, analyzer = analyze_graph_features(df, features_df)
        
        print("\n=== Graph Analysis Results ===")
        print(f"Total users: {len(features_with_graph)}")
        print(f"\nGraph metrics:")
        graph_cols = [col for col in features_with_graph.columns if col.startswith('graph_')]
        print(features_with_graph[['user_id'] + graph_cols].head())
        
        print(f"\nGraph statistics:")
        print(f"Nodes: {analyzer.graph.number_of_nodes()}")
        print(f"Edges: {analyzer.graph.number_of_edges()}")
        print(f"Density: {nx.density(analyzer.graph):.4f}")
