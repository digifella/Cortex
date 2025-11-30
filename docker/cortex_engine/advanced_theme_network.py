"""
Advanced Theme Network Visualizer - Research Rabbit Style
Creates interactive network graphs similar to Research Rabbit's author networks
with improved node positioning, sizing, and relationship visualization.

Version: 2.0.0
Date: 2025-08-04
"""

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
import math
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class AdvancedThemeNetworkVisualizer:
    """
    Creates Research Rabbit-style interactive theme network visualizations.
    Features hierarchical layouts, intelligent node sizing, and community detection.
    """
    
    def __init__(self):
        self.theme_graph = nx.Graph()
        self.theme_embeddings = {}
        self.communities = {}
        self.layout_cache = {}
        
    def create_research_rabbit_network(self, theme_data: Dict[str, Any], 
                                     layout_algorithm: str = "force_directed",
                                     max_themes: int = 50) -> go.Figure:
        """
        Create a Research Rabbit-style network visualization.
        
        Args:
            theme_data: Theme analysis data from discovery
            layout_algorithm: "force_directed", "hierarchical", or "clustered"
            max_themes: Maximum number of themes to display
            
        Returns:
            Interactive Plotly network figure
        """
        try:
            # Extract and prepare theme data
            themes = theme_data.get("themes", [])[:max_themes]
            metrics = theme_data.get("metrics", {})
            cooccurrence = theme_data.get("cooccurrence", {})
            
            if not themes:
                return self._create_empty_figure("No themes available for visualization")
            
            # Build enhanced network graph
            self._build_enhanced_graph(themes, cooccurrence, metrics)
            
            # Detect communities for color coding
            self._detect_communities()
            
            # Calculate layout positions
            layout_positions = self._calculate_layout(layout_algorithm)
            
            # Create the interactive visualization
            fig = self._create_interactive_network(
                themes, layout_positions, metrics, layout_algorithm
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Research Rabbit network creation failed: {e}")
            return self._create_empty_figure(f"Visualization error: {str(e)}")
    
    def _build_enhanced_graph(self, themes: List[Dict], cooccurrence: Dict, metrics: Dict):
        """Build NetworkX graph with enhanced node and edge attributes."""
        self.theme_graph = nx.Graph()
        
        # Add nodes with comprehensive attributes
        for theme_data in themes:
            if isinstance(theme_data, dict):
                theme_name = theme_data.get("theme", "")
                frequency = theme_data.get("frequency", 1)
                
                if theme_name:
                    # Calculate importance score
                    centrality = metrics.get("centrality", {}).get(theme_name, 0)
                    betweenness = metrics.get("betweenness", {}).get(theme_name, 0)
                    importance = (frequency * 0.4) + (centrality * 0.3) + (betweenness * 0.3)
                    
                    self.theme_graph.add_node(
                        theme_name,
                        frequency=frequency,
                        importance=importance,
                        centrality=centrality,
                        betweenness=betweenness,
                        label=theme_name
                    )
        
        # Add edges with weights and filtering
        edge_threshold = 0.15  # Only show stronger relationships
        for (theme1, theme2), weight in cooccurrence.items():
            if (weight >= edge_threshold and 
                theme1 in self.theme_graph and 
                theme2 in self.theme_graph):
                self.theme_graph.add_edge(theme1, theme2, weight=weight)
        
        logger.info(f"Enhanced graph: {self.theme_graph.number_of_nodes()} nodes, "
                   f"{self.theme_graph.number_of_edges()} edges")
    
    def _detect_communities(self):
        """Detect theme communities for color coding."""
        try:
            if self.theme_graph.number_of_nodes() < 3:
                # Too few nodes for community detection
                self.communities = {node: 0 for node in self.theme_graph.nodes()}
                return
            
            # Use Louvain community detection
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(self.theme_graph)
            
            # Map nodes to community IDs
            self.communities = {}
            for i, community in enumerate(communities):
                for node in community:
                    self.communities[node] = i
                    
            logger.info(f"Detected {len(communities)} theme communities")
            
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            # Fallback: assign all nodes to same community
            self.communities = {node: 0 for node in self.theme_graph.nodes()}
    
    def _calculate_layout(self, algorithm: str) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions using specified layout algorithm."""
        try:
            if self.theme_graph.number_of_nodes() == 0:
                return {}
            
            if algorithm == "hierarchical":
                return self._hierarchical_layout()
            elif algorithm == "clustered":
                return self._clustered_layout()
            else:  # force_directed (default)
                return self._force_directed_layout()
                
        except Exception as e:
            logger.error(f"Layout calculation failed: {e}")
            # Fallback to simple circular layout
            return nx.circular_layout(self.theme_graph)
    
    def _force_directed_layout(self) -> Dict[str, Tuple[float, float]]:
        """Enhanced force-directed layout similar to Research Rabbit."""
        # Use spring layout with careful parameter tuning
        pos = nx.spring_layout(
            self.theme_graph,
            k=3.0,  # Optimal distance between nodes
            iterations=100,  # More iterations for better layout
            weight='weight',  # Use edge weights
            scale=2.0,  # Larger scale for better separation
            center=(0, 0)
        )
        
        # Post-process to avoid overlaps
        return self._avoid_node_overlaps(pos)
    
    def _hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout based on theme importance."""
        nodes = list(self.theme_graph.nodes())
        if not nodes:
            return {}
        
        # Sort nodes by importance
        nodes.sort(key=lambda n: self.theme_graph.nodes[n].get('importance', 0), reverse=True)
        
        # Create levels based on importance
        levels = []
        nodes_per_level = max(3, int(math.sqrt(len(nodes))))
        
        for i in range(0, len(nodes), nodes_per_level):
            levels.append(nodes[i:i + nodes_per_level])
        
        # Position nodes
        pos = {}
        for level_idx, level_nodes in enumerate(levels):
            y = 2.0 - (4.0 * level_idx / max(1, len(levels) - 1))  # Top to bottom
            
            for node_idx, node in enumerate(level_nodes):
                if len(level_nodes) == 1:
                    x = 0
                else:
                    x = -2.0 + (4.0 * node_idx / (len(level_nodes) - 1))  # Left to right
                pos[node] = (x, y)
        
        return pos
    
    def _clustered_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create clustered layout grouping themes by community."""
        if not self.communities:
            return self._force_directed_layout()
        
        # Group nodes by community
        community_nodes = defaultdict(list)
        for node, community in self.communities.items():
            community_nodes[community].append(node)
        
        pos = {}
        num_communities = len(community_nodes)
        
        if num_communities == 1:
            # Single community - use circular layout
            return nx.circular_layout(self.theme_graph, scale=2.0)
        
        # Position communities in a circle
        community_positions = {}
        for i, community_id in enumerate(community_nodes.keys()):
            angle = 2 * math.pi * i / num_communities
            community_positions[community_id] = (
                2.0 * math.cos(angle),
                2.0 * math.sin(angle)
            )
        
        # Position nodes within each community
        for community_id, nodes in community_nodes.items():
            center_x, center_y = community_positions[community_id]
            
            if len(nodes) == 1:
                pos[nodes[0]] = (center_x, center_y)
            else:
                # Circular layout within community
                for i, node in enumerate(nodes):
                    angle = 2 * math.pi * i / len(nodes)
                    radius = 0.8  # Radius within community
                    x = center_x + radius * math.cos(angle)
                    y = center_y + radius * math.sin(angle)
                    pos[node] = (x, y)
        
        return pos
    
    def _avoid_node_overlaps(self, pos: Dict) -> Dict:
        """Post-process layout to avoid node overlaps."""
        min_distance = 0.3  # Minimum distance between nodes
        
        # Convert to list for easier manipulation
        nodes = list(pos.keys())
        positions = np.array([pos[node] for node in nodes])
        
        # Iteratively adjust positions to avoid overlaps
        for iteration in range(10):  # Max 10 adjustment iterations
            adjusted = False
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    
                    if dist < min_distance and dist > 0:
                        # Calculate repulsion vector
                        direction = (positions[i] - positions[j]) / dist
                        adjustment = direction * (min_distance - dist) / 2
                        
                        positions[i] += adjustment
                        positions[j] -= adjustment
                        adjusted = True
            
            if not adjusted:
                break
        
        # Convert back to dictionary
        return {nodes[i]: tuple(positions[i]) for i in range(len(nodes))}
    
    def _create_interactive_network(self, themes: List[Dict], positions: Dict, 
                                  metrics: Dict, layout_algorithm: str) -> go.Figure:
        """Create the interactive Plotly network visualization."""
        
        # Prepare node data
        node_data = self._prepare_node_data(themes, positions, metrics)
        edge_data = self._prepare_edge_data(positions)
        
        # Create edge traces
        edge_trace = go.Scatter(
            x=edge_data['x'],
            y=edge_data['y'],
            line=dict(width=edge_data['widths'], color='rgba(125, 125, 125, 0.3)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_data['x'],
            y=node_data['y'],
            mode='markers+text',
            text=node_data['labels'],
            textposition="middle center",
            textfont=dict(
                size=node_data['text_sizes'],
                color='white',
                family="Arial Black"
            ),
            marker=dict(
                size=node_data['sizes'],
                color=node_data['colors'],
                colorscale='Set3',
                opacity=0.8,
                line=dict(width=2, color='white'),
                sizemode='diameter'
            ),
            hovertemplate=node_data['hover_text'],
            hoverinfo='text',
            showlegend=False
        )
        
        # Create figure with enhanced styling
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=self._create_enhanced_layout(layout_algorithm)
        )
        
        # Add click interactivity
        fig.update_traces(
            selector=dict(mode='markers+text'),
            customdata=node_data['theme_names']
        )
        
        return fig
    
    def _prepare_node_data(self, themes: List[Dict], positions: Dict, metrics: Dict) -> Dict:
        """Prepare node data for visualization."""
        node_data = {
            'x': [], 'y': [], 'sizes': [], 'colors': [], 'labels': [],
            'hover_text': [], 'text_sizes': [], 'theme_names': []
        }
        
        # Calculate size scaling
        importances = [self.theme_graph.nodes[theme['theme']].get('importance', 1) 
                      for theme in themes if theme.get('theme') in positions]
        
        if importances:
            min_importance = min(importances)
            max_importance = max(importances)
            importance_range = max_importance - min_importance or 1
        else:
            min_importance = max_importance = importance_range = 1
        
        for theme_data in themes:
            theme_name = theme_data.get('theme', '')
            if theme_name not in positions:
                continue
            
            x, y = positions[theme_name]
            importance = self.theme_graph.nodes[theme_name].get('importance', 1)
            frequency = theme_data.get('frequency', 1)
            community = self.communities.get(theme_name, 0)
            
            # Calculate node size (15-60 pixel range)
            normalized_importance = (importance - min_importance) / importance_range
            size = 15 + (normalized_importance * 45)
            
            # Calculate text size based on node size
            text_size = max(8, min(14, int(size / 4)))
            
            # Truncate long theme names for display
            display_label = theme_name
            if len(display_label) > 20:
                display_label = display_label[:17] + "..."
            
            # Create detailed hover text
            connections = len(list(self.theme_graph.neighbors(theme_name)))
            hover_text = (
                f"<b>{theme_name}</b><br>"
                f"Frequency: {frequency}<br>"
                f"Connections: {connections}<br>"
                f"Importance: {importance:.2f}<br>"
                f"Community: {community + 1}<br>"
                "<extra></extra>"
            )
            
            node_data['x'].append(x)
            node_data['y'].append(y)
            node_data['sizes'].append(size)
            node_data['colors'].append(community)
            node_data['labels'].append(display_label)
            node_data['hover_text'].append(hover_text)
            node_data['text_sizes'].append(text_size)
            node_data['theme_names'].append(theme_name)
        
        return node_data
    
    def _prepare_edge_data(self, positions: Dict) -> Dict:
        """Prepare edge data for visualization."""
        edge_x, edge_y, widths = [], [], []
        
        for edge in self.theme_graph.edges(data=True):
            source, target, data = edge
            
            if source in positions and target in positions:
                x0, y0 = positions[source]
                x1, y1 = positions[target]
                weight = data.get('weight', 0.1)
                
                # Calculate edge width (1-8 pixel range)
                width = max(1, min(8, weight * 12))
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                widths.extend([width, width, width])
        
        return {'x': edge_x, 'y': edge_y, 'widths': widths}
    
    def _create_enhanced_layout(self, algorithm: str) -> go.Layout:
        """Create enhanced layout configuration."""
        return go.Layout(
            title=dict(
                text=f"Theme Relationship Network ({algorithm.replace('_', ' ').title()})",
                x=0.5,
                font=dict(size=18, color="#2E4057")
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            annotations=[
                dict(
                    text="💡 Hover for details • Click nodes to explore • Drag to reposition",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="#666", size=11)
                )
            ],
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False
            ),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white',
            height=700,
            dragmode='pan',
            # Enable zooming and panning
            modebar=dict(
                orientation='v',
                bgcolor='rgba(255,255,255,0.8)',
                color='#666',
                activecolor='#2E4057'
            )
        )
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="#666")
        )
        fig.update_layout(
            title="Theme Network Visualization",
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white',
            height=500
        )
        return fig
    
    def create_theme_importance_chart(self, themes: List[Dict]) -> go.Figure:
        """Create a complementary importance chart to replace histogram."""
        try:
            # Prepare data
            theme_names = []
            frequencies = []
            importances = []
            
            for theme_data in themes[:15]:  # Top 15 themes
                theme_name = theme_data.get('theme', '')
                frequency = theme_data.get('frequency', 0)
                
                if theme_name in self.theme_graph:
                    importance = self.theme_graph.nodes[theme_name].get('importance', 0)
                    theme_names.append(theme_name[:30])  # Truncate long names
                    frequencies.append(frequency)
                    importances.append(importance)
            
            if not theme_names:
                return self._create_empty_figure("No theme data available")
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            # Add importance bars
            fig.add_trace(go.Bar(
                y=theme_names,
                x=importances,
                orientation='h',
                name='Theme Importance',
                marker=dict(
                    color=importances,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=[f'{imp:.1f}' for imp in importances],
                textposition='inside',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Theme Importance Ranking',
                xaxis_title='Importance Score',
                yaxis_title='Themes',
                height=max(400, len(theme_names) * 25),
                plot_bgcolor='#FAFAFA',
                paper_bgcolor='white',
                font=dict(size=11)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Importance chart creation failed: {e}")
            return self._create_empty_figure("Chart creation error")
    
    def get_layout_options(self) -> List[Dict]:
        """Get available layout algorithm options."""
        return [
            {
                "value": "force_directed",
                "label": "Force-Directed Layout",
                "description": "Natural clustering based on theme relationships"
            },
            {
                "value": "hierarchical", 
                "label": "Hierarchical Layout",
                "description": "Arranged by theme importance (top-down)"
            },
            {
                "value": "clustered",
                "label": "Community Clusters", 
                "description": "Grouped by detected theme communities"
            }
        ]