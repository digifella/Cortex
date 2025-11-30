"""
Theme Visualization Module for Idea Generator

This module provides interactive network visualization of themes and their relationships,
similar to Research Rabbit's author networks. It creates hoverable, interactive graphs
that help users explore theme connections and select focus areas for ideation.

Version: 1.0.0
Date: 2025-08-03
"""

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
import logging
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ThemeNetworkVisualizer:
    """
    Creates interactive network visualizations of theme relationships from discovery data.
    """
    
    def __init__(self):
        """Initialize the theme visualizer."""
        self.theme_graph = nx.Graph()
        self.theme_data = {}
        self.document_themes = {}
        logger.info("ThemeNetworkVisualizer initialized")
    
    def extract_themes_from_discovery(self, discovery_results: Dict[str, Any], 
                                    collection_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract and analyze themes from discovery results and collection content.
        
        Args:
            discovery_results: Results from the discovery phase
            collection_content: Raw collection documents
            
        Returns:
            Dict containing theme analysis and network data
        """
        try:
            logger.info("Extracting substantial business themes from content analysis")
            
            # Extract themes from discovery results (these are LLM-generated and should be better)
            discovery_themes = discovery_results.get("themes", [])
            opportunities = discovery_results.get("opportunities", [])
            key_insights = discovery_results.get("key_insights", [])
            
            # PRIMARY: Use intelligent content analysis for business themes
            content_themes = self._extract_strategic_content_themes(collection_content)
            
            # SECONDARY: Use metadata themes only as supplement (they tend to be administrative)
            metadata_themes = self._extract_metadata_themes(collection_content) if len(content_themes) < 8 else []
            
            # Combine with priority on content themes and discovery results
            all_themes = self._normalize_themes(content_themes + discovery_themes + opportunities + key_insights + metadata_themes)
            
            # Calculate theme co-occurrence
            theme_cooccurrence = self._calculate_theme_cooccurrence(all_themes, collection_content)
            
            # Build theme relationship graph
            self._build_theme_graph(all_themes, theme_cooccurrence)
            
            # Calculate theme metrics
            theme_metrics = self._calculate_theme_metrics(all_themes, collection_content)
            
            return {
                "themes": all_themes,
                "cooccurrence": theme_cooccurrence,
                "metrics": theme_metrics,
                "graph_stats": self._get_graph_statistics(),
                "visualization_data": self._prepare_visualization_data()
            }
            
        except Exception as e:
            logger.error(f"Theme extraction failed: {e}")
            return {"error": str(e)}
    
    def _extract_metadata_themes(self, collection_content: List[Dict[str, Any]]) -> List[str]:
        """
        Extract themes from document metadata thematic_tags.
        This provides fundamental business themes like PMO, strategy, change management, etc.
        
        Args:
            collection_content: List of documents with metadata
            
        Returns:
            List of themes with frequency counts as tuples (theme, count)
        """
        try:
            if not collection_content:
                return []
            
            # Collect all thematic tags from metadata
            theme_counts = Counter()
            
            for doc in collection_content:
                metadata = doc.get('metadata', {})
                
                # Check for thematic_tags in metadata
                thematic_tags = metadata.get('thematic_tags', [])
                
                # Handle different formats (string vs list)
                if isinstance(thematic_tags, str):
                    # Split comma-separated string
                    tags = [tag.strip() for tag in thematic_tags.split(',') if tag.strip()]
                elif isinstance(thematic_tags, list):
                    tags = [str(tag).strip() for tag in thematic_tags if str(tag).strip()]
                else:
                    tags = []
                
                # Clean and count themes
                for tag in tags:
                    if tag and len(tag) > 1:  # Skip empty or single character tags
                        cleaned_tag = self._clean_theme_text(tag)
                        if cleaned_tag:
                            theme_counts[cleaned_tag] += 1
            
            # Convert to list of tuples (theme, frequency) sorted by frequency
            themes_with_frequency = []
            # Limit themes to most relevant ones for ideation
            max_themes = min(50, len(theme_counts))  # Up to 50 themes maximum
            
            # Increase minimum frequency to focus on more substantial themes
            collection_size = len(collection_content)
            if collection_size > 2000:
                min_frequency = max(3, collection_size // 500)  # At least 3, or 1 per 500 docs
            elif collection_size > 500:
                min_frequency = max(2, collection_size // 250)  # At least 2, or 1 per 250 docs
            else:
                min_frequency = max(1, collection_size // 100)  # At least 1, or 1 per 100 docs
            
            for theme, count in theme_counts.most_common(max_themes):
                # Filter out low frequency themes to focus on substantial patterns
                if count >= min_frequency:
                    themes_with_frequency.append({
                        "theme": theme,
                        "frequency": count
                    })
            
            logger.info(f"Extracted {len(themes_with_frequency)} themes from metadata after filtering")
            if themes_with_frequency:
                logger.info(f"Top themes: {[t['theme'] for t in themes_with_frequency[:5]]}")
            else:
                logger.warning("No themes found after filtering - this suggests an issue with metadata extraction")
            
            return themes_with_frequency
            
        except Exception as e:
            logger.error(f"Metadata theme extraction failed: {e}")
            return []
    
    def _extract_strategic_content_themes(self, collection_content: List[Dict[str, Any]], 
                                         max_features: int = 15) -> List[str]:
        """
        Extract strategic business themes from document content using intelligent analysis.
        Focuses on substantial business concepts, technologies, and domain-specific themes.
        """
        try:
            if not collection_content:
                return []
            
            # Combine all document content with emphasis on titles and key sections
            documents = []
            for doc in collection_content:
                content = doc.get('content', '')
                title = doc.get('title', '')
                doc_type = doc.get('document_type', '')
                
                # Weight different parts differently - titles and types are more indicative
                weighted_content = f"{title} {title} {title} {doc_type} {doc_type} {content}"
                documents.append(weighted_content)
            
            if not documents:
                return []
            
            # Use TF-IDF with strategic focus - prefer longer, meaningful phrases
            vectorizer = TfidfVectorizer(
                max_features=max_features * 3,  # Get more candidates to filter
                stop_words='english',
                ngram_range=(2, 4),  # Focus on 2-4 word phrases (strategic concepts)
                min_df=2,  # Must appear in at least 2 documents
                max_df=0.7,  # Don't use terms that appear in most documents
                token_pattern=r'\b[A-Za-z]{3,}\b',  # Only words 3+ characters
            )
            
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores for each feature
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Create theme candidates with scores
            theme_candidates = list(zip(feature_names, mean_scores))
            theme_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Filter for strategic business themes
            strategic_themes = []
            business_indicators = {
                'technology', 'digital', 'innovation', 'strategy', 'transformation',
                'implementation', 'development', 'improvement', 'optimization',
                'analysis', 'assessment', 'evaluation', 'framework', 'methodology',
                'solution', 'system', 'platform', 'architecture', 'design',
                'security', 'risk', 'compliance', 'governance', 'policy',
                'customer', 'client', 'service', 'delivery', 'performance',
                'quality', 'efficiency', 'effectiveness', 'capability', 'maturity',
                'integration', 'migration', 'modernization', 'automation',
                'data', 'analytics', 'intelligence', 'insights', 'reporting',
                'cloud', 'infrastructure', 'network', 'application', 'software'
            }
            
            for theme, score in theme_candidates:
                cleaned_theme = self._clean_theme_text(theme)
                if cleaned_theme and len(cleaned_theme.split()) >= 2:
                    # Check if theme contains business/strategic indicators
                    theme_words = cleaned_theme.lower().split()
                    has_business_context = any(word in business_indicators for word in theme_words)
                    
                    # Accept if it has business context or is a substantial multi-word concept
                    if has_business_context or (len(cleaned_theme.split()) >= 3 and score > 0.1):
                        strategic_themes.append({
                            "theme": cleaned_theme,
                            "frequency": int(score * 100)  # Convert score to frequency-like metric
                        })
                        
                        if len(strategic_themes) >= max_features:
                            break
            
            logger.info(f"Extracted {len(strategic_themes)} strategic content themes")
            if strategic_themes:
                logger.info(f"Top strategic themes: {[t['theme'] for t in strategic_themes[:5]]}")
            
            return strategic_themes
            
        except Exception as e:
            logger.error(f"Strategic content theme extraction failed: {e}")
            return []

    def _extract_content_themes(self, collection_content: List[Dict[str, Any]], 
                               max_features: int = 20) -> List[str]:
        """
        Extract key themes from document content using TF-IDF analysis.
        
        Args:
            collection_content: List of documents with content
            max_features: Maximum number of theme features to extract
            
        Returns:
            List of extracted theme strings
        """
        try:
            if not collection_content:
                return []
            
            # Combine all document content
            documents = []
            for doc in collection_content:
                content = doc.get('content', '')
                title = doc.get('title', '')
                # Combine title and content, giving title more weight
                combined = f"{title} {title} {content}"
                documents.append(combined)
            
            if not documents:
                return []
            
            # Use TF-IDF to extract key terms
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 3),  # Include 1-3 word phrases
                min_df=2,  # Term must appear in at least 2 documents
                max_df=0.8  # Ignore terms that appear in more than 80% of documents
            )
            
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores for each feature
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Create theme list with scores
            theme_scores = list(zip(feature_names, mean_scores))
            theme_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top themes, cleaned up
            themes = []
            for theme, score in theme_scores[:max_features]:
                # Clean up theme text
                cleaned_theme = self._clean_theme_text(theme)
                if cleaned_theme and len(cleaned_theme) > 2:
                    themes.append(cleaned_theme)
            
            logger.info(f"Extracted {len(themes)} content themes")
            return themes
            
        except Exception as e:
            logger.error(f"Content theme extraction failed: {e}")
            return []
    
    def _clean_theme_text(self, theme: str) -> str:
        """Clean and normalize theme text, filtering out generic and organizational terms."""
        if not theme or not isinstance(theme, str):
            return ""
            
        # Convert to lowercase for filtering
        theme_lower = theme.lower().strip()
        
        # Filter out technical file formats and irrelevant terms
        technical_terms = {
            'pdf', 'docx', 'doc', 'xlsx', 'xls', 'ppt', 'pptx', 'txt', 'csv',
            'jpg', 'jpeg', 'png', 'gif', 'tiff', 'bmp', 'zip', 'rar', '7z',
            'exe', 'dll', 'sys', 'bat', 'cmd', 'ps1', 'sh', 'py', 'java',
            'html', 'css', 'js', 'xml', 'json', 'sql', 'log', 'tmp'
        }
        
        # Filter out generic business/meeting terms that add no thematic value
        generic_terms = {
            # Meeting and administrative terms
            'meeting', 'meetings', 'conference', 'conferences', 'collaboration', 
            'collaborations', 'presentation', 'presentations', 'discussion', 
            'discussions', 'session', 'sessions', 'workshop', 'workshops',
            'document', 'documents', 'report', 'reports', 'file', 'files',
            'email', 'emails', 'communication', 'communications', 'update',
            'updates', 'review', 'reviews', 'minutes', 'agenda', 'agendas',
            'action', 'actions', 'item', 'items', 'team', 'teams', 'group',
            'groups', 'member', 'members', 'staff', 'people', 'person',
            'date', 'dates', 'time', 'times', 'week', 'weeks', 'month',
            'months', 'year', 'years', 'day', 'days', 'hour', 'hours',
            
            # Process-focused terms that don't indicate business themes
            'meeting minutes', 'public speaking', 'professional gathering',
            'office environment', 'workplace', 'administrative', 'coordination',
            'scheduling', 'planning', 'organizing', 'documentation', 'recording',
            'tracking', 'monitoring', 'reporting', 'briefing', 'debriefing',
            'attendance', 'participation', 'engagement', 'networking',
            'communication skills', 'presentation skills', 'interpersonal',
            'teamwork', 'cooperation', 'procedure', 'process', 'protocol',
            'guidelines', 'standards', 'compliance', 'governance', 'oversight',
            'administration', 'management', 'supervision', 'leadership',
            'project management', 'time management', 'resource management'
        }
        
        # Common organizational terms that are usually client/company names
        organizational_terms = {
            'cenitex', 'government', 'dept', 'department', 'division', 'unit',
            'branch', 'section', 'office', 'bureau', 'agency', 'authority',
            'council', 'board', 'committee', 'commission', 'corporation',
            'company', 'enterprise', 'organization', 'organisation', 'firm',
            'business', 'service', 'services', 'group', 'limited', 'ltd',
            'inc', 'incorporated', 'pty', 'proprietary'
        }
        
        # Check if it's a filtered term
        if (theme_lower in technical_terms or 
            theme_lower in generic_terms or 
            theme_lower in organizational_terms):
            return ""
        
        # Filter out very short themes (prefer meaningful multi-word themes)
        if len(theme_lower) <= 3:
            return ""
        
        # Prefer multi-word themes over single words
        word_count = len(theme_lower.split())
        if word_count == 1 and len(theme_lower) < 8:
            # Single short words are usually too generic
            return ""
            
        # Remove special characters, normalize spacing
        cleaned = re.sub(r'[^\w\s]', ' ', theme)
        cleaned = ' '.join(cleaned.split())
        
        # Final check: ensure it's not just numbers or common words
        if cleaned.lower() in {'new', 'old', 'current', 'previous', 'next', 'last', 'first', 'second', 'third'}:
            return ""
        
        # Return in title case for consistency
        return cleaned.title()
    
    def _normalize_themes(self, themes: List[str]) -> List[str]:
        """Normalize and deduplicate theme list, handling both strings and dictionaries."""
        normalized = []
        seen_themes = set()
        
        for theme in themes:
            if isinstance(theme, dict):
                # Handle theme with frequency data
                theme_name = theme.get("theme", "")
                frequency = theme.get("frequency", 0)
                if theme_name and theme_name.strip():
                    cleaned = self._clean_theme_text(theme_name.strip())
                    if len(cleaned) > 2 and cleaned not in seen_themes:
                        normalized.append({
                            "theme": cleaned,
                            "frequency": frequency
                        })
                        seen_themes.add(cleaned)
            elif isinstance(theme, str) and theme.strip():
                # Handle plain string themes
                cleaned = self._clean_theme_text(theme.strip())
                if len(cleaned) > 2 and cleaned not in seen_themes:
                    normalized.append({
                        "theme": cleaned,
                        "frequency": 1  # Default frequency for string themes
                    })
                    seen_themes.add(cleaned)
        
        # Sort by frequency (highest first)
        normalized.sort(key=lambda x: x.get("frequency", 0), reverse=True)
        
        return normalized
    
    def _calculate_theme_cooccurrence(self, themes: List[Dict[str, Any]], 
                                    collection_content: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
        """
        Calculate co-occurrence strength between themes based on document content and metadata.
        
        Args:
            themes: List of theme dictionaries with 'theme' and 'frequency' keys
            collection_content: Document collection
            
        Returns:
            Dict mapping theme pairs to co-occurrence strength
        """
        try:
            cooccurrence = defaultdict(float)
            
            # Extract theme names for easier processing
            theme_names = [theme.get("theme", "") for theme in themes if isinstance(theme, dict)]
            
            # For each document, find which themes appear
            for doc in collection_content:
                content = doc.get('content', '').lower()
                title = doc.get('title', '').lower()
                metadata = doc.get('metadata', {})
                
                # Get thematic tags from metadata first
                metadata_tags = metadata.get('thematic_tags', [])
                if isinstance(metadata_tags, str):
                    metadata_tags = [tag.strip().lower() for tag in metadata_tags.split(',') if tag.strip()]
                else:
                    metadata_tags = [str(tag).lower() for tag in metadata_tags if tag]
                
                combined_text = f"{title} {content}"
                
                # Find themes present in this document
                doc_themes = []
                for theme_name in theme_names:
                    if not theme_name:
                        continue
                        
                    theme_lower = theme_name.lower()
                    
                    # Priority check: metadata tags (exact match)
                    if any(theme_lower in tag or tag in theme_lower for tag in metadata_tags):
                        doc_themes.append(theme_name)
                    # Fallback: content text search (fuzzy matching)
                    elif (theme_lower in combined_text or 
                          any(word in combined_text for word in theme_lower.split())):
                        doc_themes.append(theme_name)
                
                # Calculate co-occurrence for theme pairs in this document
                for i, theme1 in enumerate(doc_themes):
                    for j, theme2 in enumerate(doc_themes):
                        if i < j:  # Avoid duplicates and self-references
                            pair = (theme1, theme2)
                            cooccurrence[pair] += 1.0
            
            # Normalize co-occurrence scores
            max_cooccurrence = max(cooccurrence.values()) if cooccurrence else 1
            normalized_cooccurrence = {
                pair: score / max_cooccurrence 
                for pair, score in cooccurrence.items()
            }
            
            logger.info(f"Calculated co-occurrence for {len(normalized_cooccurrence)} theme pairs")
            return normalized_cooccurrence
            
        except Exception as e:
            logger.error(f"Co-occurrence calculation failed: {e}")
            return {}
    
    def _build_theme_graph(self, themes: List[Dict[str, Any]], 
                          cooccurrence: Dict[Tuple[str, str], float]):
        """Build NetworkX graph from themes and co-occurrence data."""
        try:
            self.theme_graph = nx.Graph()
            
            # Add theme nodes with frequency data
            for theme_data in themes:
                if isinstance(theme_data, dict):
                    theme_name = theme_data.get("theme", "")
                    frequency = theme_data.get("frequency", 0)
                    if theme_name:
                        self.theme_graph.add_node(theme_name, 
                                                theme_name=theme_name, 
                                                frequency=frequency)
            
            # Add edges based on co-occurrence
            min_weight = 0.1  # Minimum edge weight threshold
            for (theme1, theme2), weight in cooccurrence.items():
                if weight >= min_weight:
                    self.theme_graph.add_edge(theme1, theme2, weight=weight)
            
            logger.info(f"Built theme graph with {self.theme_graph.number_of_nodes()} nodes and {self.theme_graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Theme graph building failed: {e}")
    
    def _calculate_theme_metrics(self, themes: List[Dict[str, Any]], 
                               collection_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate various metrics for themes."""
        try:
            metrics = {}
            
            # Calculate centrality measures
            if self.theme_graph.number_of_nodes() > 0:
                centrality = nx.degree_centrality(self.theme_graph)
                betweenness = nx.betweenness_centrality(self.theme_graph)
                closeness = nx.closeness_centrality(self.theme_graph)
                
                metrics["centrality"] = centrality
                metrics["betweenness"] = betweenness
                metrics["closeness"] = closeness
            
            # Use frequency data from themes directly (from metadata)
            theme_frequency = {}
            for theme_data in themes:
                if isinstance(theme_data, dict):
                    theme_name = theme_data.get("theme", "")
                    frequency = theme_data.get("frequency", 0)
                    if theme_name:
                        theme_frequency[theme_name] = frequency
            
            metrics["frequency"] = dict(theme_frequency)
            
            # Identify clusters/communities
            if self.theme_graph.number_of_nodes() > 2:
                try:
                    communities = nx.community.greedy_modularity_communities(self.theme_graph)
                    community_map = {}
                    for i, community in enumerate(communities):
                        for node in community:
                            community_map[node] = i
                    metrics["communities"] = community_map
                except Exception as e:
                    logger.debug(f"Community detection failed: {e}")
                    metrics["communities"] = {}
            
            return metrics
            
        except Exception as e:
            logger.error(f"Theme metrics calculation failed: {e}")
            return {}
    
    def _get_graph_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the theme graph."""
        if self.theme_graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0, "density": 0, "components": 0}
        
        return {
            "nodes": self.theme_graph.number_of_nodes(),
            "edges": self.theme_graph.number_of_edges(),
            "density": nx.density(self.theme_graph),
            "components": nx.number_connected_components(self.theme_graph),
            "average_clustering": nx.average_clustering(self.theme_graph)
        }
    
    def _prepare_visualization_data(self) -> Dict[str, Any]:
        """Prepare data for Plotly visualization."""
        try:
            if self.theme_graph.number_of_nodes() == 0:
                return {"nodes": [], "edges": []}
            
            # Calculate layout using spring layout
            pos = nx.spring_layout(self.theme_graph, k=3, iterations=50)
            
            # Prepare node data
            nodes = []
            for node in self.theme_graph.nodes():
                x, y = pos[node]
                degree = self.theme_graph.degree(node)
                
                nodes.append({
                    "id": node,
                    "label": node,
                    "x": x,
                    "y": y,
                    "degree": degree,
                    "size": max(10, min(50, degree * 5 + 10))  # Size based on degree
                })
            
            # Prepare edge data
            edges = []
            for edge in self.theme_graph.edges(data=True):
                source, target, data = edge
                weight = data.get('weight', 0.1)
                
                edges.append({
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "width": max(1, weight * 5)  # Width based on weight
                })
            
            return {"nodes": nodes, "edges": edges, "layout": pos}
            
        except Exception as e:
            logger.error(f"Visualization data preparation failed: {e}")
            return {"nodes": [], "edges": []}
    
    def create_interactive_network(self, theme_data: Dict[str, Any], 
                                 title: str = "Theme Relationship Network") -> go.Figure:
        """
        Create interactive Plotly network visualization.
        
        Args:
            theme_data: Theme analysis data from extract_themes_from_discovery
            title: Title for the visualization
            
        Returns:
            Plotly Figure object
        """
        try:
            viz_data = theme_data.get("visualization_data", {})
            nodes = viz_data.get("nodes", [])
            edges = viz_data.get("edges", [])
            
            if not nodes:
                # Create empty plot
                fig = go.Figure()
                fig.add_annotation(
                    text="No themes found for visualization",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                fig.update_layout(title=title)
                return fig
            
            # Create edge traces
            edge_x, edge_y = [], []
            edge_info = []
            
            for edge in edges:
                source_node = next((n for n in nodes if n["id"] == edge["source"]), None)
                target_node = next((n for n in nodes if n["id"] == edge["target"]), None)
                
                if source_node and target_node:
                    edge_x.extend([source_node["x"], target_node["x"], None])
                    edge_y.extend([source_node["y"], target_node["y"], None])
                    edge_info.append(f"{edge['source']} ↔ {edge['target']} (weight: {edge['weight']:.2f})")
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create node trace
            node_x = [node["x"] for node in nodes]
            node_y = [node["y"] for node in nodes]
            node_text = [node["label"] for node in nodes]
            node_sizes = [node["size"] for node in nodes]
            node_degrees = [node["degree"] for node in nodes]
            
            # Create hover text with detailed information
            hover_text = []
            for node in nodes:
                hover_info = f"<b>{node['label']}</b><br>"
                hover_info += f"Connections: {node['degree']}<br>"
                hover_info += f"Click to explore this theme<br>"
                hover_info += "<extra></extra>"  # Hide default hover box
                hover_text.append(hover_info)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hovertemplate=hover_text,
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=node_sizes,
                    color=node_degrees,
                    colorscale='Viridis',
                    colorbar=dict(
                        title="Theme Connections",
                        titleside="right"
                    ),
                    line=dict(width=2, color='white'),
                    showscale=True
                ),
                textfont=dict(size=10, color='white'),
                name="Themes"
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=dict(text=title, x=0.5),
                              titlefont_size=16,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="Click on themes to explore • Hover for details",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor='left', yanchor='bottom',
                                  font=dict(color="gray", size=12)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              plot_bgcolor='white',
                              height=600
                          ))
            
            return fig
            
        except Exception as e:
            logger.error(f"Interactive network creation failed: {e}")
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Visualization error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=14, color="red")
            )
            fig.update_layout(title=title)
            return fig
    
    def get_theme_details(self, theme: str, theme_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed information about a specific theme.
        
        Args:
            theme: Theme name to analyze
            theme_data: Theme analysis data
            
        Returns:
            Dict containing detailed theme information
        """
        try:
            metrics = theme_data.get("metrics", {})
            
            details = {
                "theme_name": theme,
                "connections": [],
                "centrality_score": metrics.get("centrality", {}).get(theme, 0),
                "frequency": metrics.get("frequency", {}).get(theme, 0),
                "community": metrics.get("communities", {}).get(theme, 0)
            }
            
            # Get connected themes
            if hasattr(self, 'theme_graph') and theme in self.theme_graph:
                neighbors = list(self.theme_graph.neighbors(theme))
                for neighbor in neighbors:
                    edge_data = self.theme_graph.edges[theme, neighbor]
                    details["connections"].append({
                        "theme": neighbor,
                        "strength": edge_data.get("weight", 0)
                    })
                
                # Sort by connection strength
                details["connections"].sort(key=lambda x: x["strength"], reverse=True)
            
            return details
            
        except Exception as e:
            logger.error(f"Theme details retrieval failed: {e}")
            return {"theme_name": theme, "error": str(e)}