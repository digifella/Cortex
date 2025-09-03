"""
Knowledge Analytics Dashboard - Comprehensive analytics and insights for the Cortex Suite
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import json
from pathlib import Path
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import logging

# Import Cortex components
from cortex_engine.config_manager import ConfigManager
from cortex_engine.graph_manager import EnhancedGraphManager
from cortex_engine.utils.logging_utils import get_logger
from cortex_engine.utils.path_utils import convert_to_docker_mount_path

# Configure logging
logger = get_logger(__name__)

st.set_page_config(
    page_title="Knowledge Analytics",
    page_icon="📊",
    layout="wide"
)

class KnowledgeAnalytics:
    """Analytics engine for knowledge base insights and metrics"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.db_path = convert_to_docker_mount_path(self.config.get('ai_database_path', '/mnt/f/ai_databases'))
        self.project_root = Path(__file__).parent.parent
        self.graph_manager = None
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize graph manager"""
        try:
            graph_path = os.path.join(self.db_path, "knowledge_cortex.gpickle")
            if os.path.exists(graph_path):
                self.graph_manager = EnhancedGraphManager(graph_path)
                logger.info("Analytics engines initialized successfully")
            else:
                logger.warning(f"Knowledge graph not found at {graph_path}")
        except Exception as e:
            logger.error(f"Failed to initialize analytics engines: {e}")
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get comprehensive document statistics"""
        stats = {
            'total_documents': 0,
            'total_entities': 0,
            'total_relationships': 0,
            'document_types': {},
            'ingestion_timeline': [],
            'collection_stats': {}
        }
        
        try:
            if self.graph_manager and self.graph_manager.graph:
                graph = self.graph_manager.graph
                
                # Count documents, entities, relationships
                documents = [n for n, d in graph.nodes(data=True) if d.get('entity_type') == 'Document']
                entities = [n for n, d in graph.nodes(data=True) if d.get('entity_type') in ['person', 'organization', 'project']]
                
                stats['total_documents'] = len(documents)
                stats['total_entities'] = len(entities)
                stats['total_relationships'] = graph.number_of_edges()
                
                # Document types breakdown
                doc_types = defaultdict(int)
                for doc in documents:
                    doc_data = graph.nodes[doc]
                    doc_type = doc_data.get('document_type', 'Unknown')
                    doc_types[doc_type] += 1
                stats['document_types'] = dict(doc_types)
                
                # Ingestion timeline (from logs)
                stats['ingestion_timeline'] = self._parse_ingestion_timeline()
                
            # Collection statistics
            stats['collection_stats'] = self._get_collection_stats()
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            
        return stats
    
    def _parse_ingestion_timeline(self) -> List[Dict]:
        """Parse ingestion log for timeline data"""
        timeline = []
        log_path = os.path.join(self.project_root, "logs", "ingestion.log")
        
        try:
            if os.path.exists(log_path):
                logger.info(f"Reading ingestion log from: {log_path}")
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    logger.info(f"Found {len(lines)} lines in ingestion log")
                    
                    for line in lines:
                        if "Analysis complete" in line:
                            # Extract timestamp and document count
                            parts = line.split(" - ")
                            if len(parts) >= 3:
                                timestamp_str = parts[0]
                                try:
                                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                                    # Extract document count from message
                                    message = parts[2]
                                    if "documents written" in message:
                                        count = int(message.split()[0])
                                        timeline.append({
                                            'timestamp': timestamp.isoformat(),
                                            'documents_processed': count
                                        })
                                        logger.info(f"Found timeline entry: {count} documents at {timestamp}")
                                except ValueError as e:
                                    logger.warning(f"Failed to parse timestamp: {timestamp_str}, error: {e}")
                                    continue
                        elif "Finalization complete" in line:
                            # Also look for finalization messages as they indicate successful processing
                            parts = line.split(" - ")
                            if len(parts) >= 3:
                                timestamp_str = parts[0]
                                try:
                                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                                    # Get the number of documents from the line before or context
                                    timeline.append({
                                        'timestamp': timestamp.isoformat(),
                                        'documents_processed': 1  # Default to 1 if we can't extract count
                                    })
                                    logger.info(f"Found finalization entry at {timestamp}")
                                except ValueError:
                                    continue
                logger.info(f"Parsed {len(timeline)} timeline entries")
            else:
                logger.warning(f"Ingestion log not found at: {log_path}")
        except Exception as e:
            logger.error(f"Error parsing ingestion timeline: {e}")
            
        return timeline
    
    def _get_collection_stats(self) -> Dict[str, Any]:
        """Get collection usage and effectiveness statistics"""
        collection_stats = {
            'total_collections': 0,
            'collection_sizes': {},
            'collection_overlap': {}
        }
        
        try:
            collections_path = os.path.join(self.project_root, "working_collections.json")
            if os.path.exists(collections_path):
                with open(collections_path, 'r') as f:
                    collections = json.load(f)
                
                collection_stats['total_collections'] = len(collections)
                
                # Collection sizes
                for name, data in collections.items():
                    collection_stats['collection_sizes'][name] = len(data.get('doc_ids', []))
                
                # Collection overlap analysis
                collection_stats['collection_overlap'] = self._calculate_collection_overlap(collections)
                
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            
        return collection_stats
    
    def _calculate_collection_overlap(self, collections: Dict) -> Dict[str, float]:
        """Calculate overlap percentages between collections"""
        overlap_matrix = {}
        collection_names = list(collections.keys())
        
        for i, name1 in enumerate(collection_names):
            for name2 in collection_names[i+1:]:
                docs1 = set(collections[name1].get('doc_ids', []))
                docs2 = set(collections[name2].get('doc_ids', []))
                
                if docs1 and docs2:
                    intersection = len(docs1.intersection(docs2))
                    union = len(docs1.union(docs2))
                    overlap_pct = (intersection / union) * 100 if union > 0 else 0
                    overlap_matrix[f"{name1} ↔ {name2}"] = overlap_pct
                    
        return overlap_matrix
    
    def get_entity_analysis(self) -> Dict[str, Any]:
        """Analyze entity distribution and relationships"""
        analysis = {
            'entity_types': {},
            'top_entities': {},
            'relationship_patterns': {},
            'entity_clusters': []
        }
        
        try:
            if self.graph_manager and self.graph_manager.graph:
                graph = self.graph_manager.graph
                
                # Entity type distribution
                entity_types = defaultdict(int)
                all_entities = {}
                
                for node, data in graph.nodes(data=True):
                    node_type = data.get('entity_type')
                    if node_type in ['person', 'organization', 'project']:
                        entity_types[node_type] += 1
                        if node_type not in all_entities:
                            all_entities[node_type] = []
                        all_entities[node_type].append((node, graph.degree(node)))
                
                analysis['entity_types'] = dict(entity_types)
                
                # Top entities by connection count
                for entity_type, entities in all_entities.items():
                    # Sort by degree (connection count) and take top 10
                    top_entities = sorted(entities, key=lambda x: x[1], reverse=True)[:10]
                    analysis['top_entities'][entity_type] = [
                        {'name': name, 'connections': degree} for name, degree in top_entities
                    ]
                
                # Relationship pattern analysis
                relationship_types = defaultdict(int)
                for u, v, data in graph.edges(data=True):
                    rel_type = data.get('relationship', 'unknown')
                    relationship_types[rel_type] += 1
                
                analysis['relationship_patterns'] = dict(relationship_types)
                
                # Entity clustering (find communities)
                analysis['entity_clusters'] = self._find_entity_clusters(graph)
                
        except Exception as e:
            logger.error(f"Error in entity analysis: {e}")
            
        return analysis
    
    def _find_entity_clusters(self, graph: nx.Graph) -> List[Dict]:
        """Find entity clusters using community detection"""
        clusters = []
        
        try:
            # Create subgraph with only entities (exclude documents)
            entity_nodes = [n for n, d in graph.nodes(data=True) 
                          if d.get('entity_type') in ['person', 'organization', 'project']]
            entity_subgraph = graph.subgraph(entity_nodes)
            
            if len(entity_subgraph.nodes()) > 2:
                # Convert to undirected graph for clustering
                undirected_subgraph = entity_subgraph.to_undirected() if entity_subgraph.is_directed() else entity_subgraph
                # Use simple connected components for clustering
                components = list(nx.connected_components(undirected_subgraph))
                
                for i, component in enumerate(components):
                    if len(component) >= 3:  # Only show clusters with 3+ entities
                        cluster_data = {
                            'cluster_id': i + 1,
                            'size': len(component),
                            'entities': []
                        }
                        
                        for entity in component:
                            entity_data = graph.nodes[entity]
                            cluster_data['entities'].append({
                                'name': entity,
                                'type': entity_data.get('entity_type', 'unknown'),
                                'connections': entity_subgraph.degree(entity)
                            })
                        
                        clusters.append(cluster_data)
                
                # Sort clusters by size
                clusters.sort(key=lambda x: x['size'], reverse=True)
                
        except Exception as e:
            logger.error(f"Error finding entity clusters: {e}")
            
        return clusters[:10]  # Return top 10 clusters
    
    def identify_knowledge_gaps(self) -> Dict[str, Any]:
        """Identify potential knowledge gaps and opportunities"""
        gaps = {
            'isolated_entities': [],
            'sparse_document_types': [],
            'low_connectivity_areas': [],
            'temporal_gaps': [],
            'recommended_additions': []
        }
        
        try:
            if self.graph_manager and self.graph_manager.graph:
                graph = self.graph_manager.graph
                
                # Find isolated entities (low connectivity)
                for node, data in graph.nodes(data=True):
                    if data.get('entity_type') in ['person', 'organization', 'project']:
                        degree = graph.degree(node)
                        if degree <= 2:  # Weakly connected
                            gaps['isolated_entities'].append({
                                'name': node,
                                'type': data.get('entity_type'),
                                'connections': degree
                            })
                
                # Analyze document type distribution for gaps
                doc_types = defaultdict(int)
                for node, data in graph.nodes(data=True):
                    if data.get('entity_type') == 'Document':
                        doc_type = data.get('document_type', 'Unknown')
                        doc_types[doc_type] += 1
                
                # Identify sparse document types
                total_docs = sum(doc_types.values())
                for doc_type, count in doc_types.items():
                    percentage = (count / total_docs) * 100 if total_docs > 0 else 0
                    if percentage < 10:  # Less than 10% representation
                        gaps['sparse_document_types'].append({
                            'type': doc_type,
                            'count': count,
                            'percentage': percentage
                        })
                
                # Generate recommendations
                gaps['recommended_additions'] = self._generate_recommendations(gaps)
                
        except Exception as e:
            logger.error(f"Error identifying knowledge gaps: {e}")
            
        return gaps
    
    def _generate_recommendations(self, gaps: Dict) -> List[str]:
        """Generate actionable recommendations based on identified gaps"""
        recommendations = []
        
        # Recommendations based on isolated entities
        if gaps['isolated_entities']:
            recommendations.append(
                f"Consider adding more documents that reference {len(gaps['isolated_entities'])} "
                "weakly connected entities to strengthen knowledge relationships."
            )
        
        # Recommendations based on sparse document types
        if gaps['sparse_document_types']:
            sparse_types = [g['type'] for g in gaps['sparse_document_types']]
            recommendations.append(
                f"Expand coverage in document types: {', '.join(sparse_types)} "
                "to achieve more balanced knowledge representation."
            )
        
        # General recommendations
        recommendations.extend([
            "Regular ingestion of new documents will help maintain knowledge currency.",
            "Review collection effectiveness and consider consolidating overlapping collections.",
            "Focus on documenting relationships between isolated entities and main knowledge clusters."
        ])
        
        return recommendations

def main():
    """Main dashboard interface"""
    st.title("📊 Knowledge Analytics Dashboard")
    st.markdown("Comprehensive insights and analytics for your Cortex Suite knowledge base")
    
    # Initialize configuration
    try:
        config_manager = ConfigManager()
        analytics = KnowledgeAnalytics(config_manager)
    except Exception as e:
        st.error(f"Failed to initialize analytics: {e}")
        return
    
    # Check if knowledge base exists
    if not analytics.graph_manager:
        st.warning("⚠️ No knowledge base found. Please run document ingestion first.")
        return
    
    # Sidebar controls
    st.sidebar.header("📊 Analytics Controls")
    
    # Analytics sections
    analysis_sections = [
        "📈 Overview Dashboard",
        "📚 Document Analytics", 
        "🕸️ Entity Network Analysis",
        "📋 Collection Insights",
        "🔍 Knowledge Gap Analysis",
        "💡 Recommendations"
    ]
    
    selected_section = st.sidebar.radio(
        "Select Analysis Section:",
        analysis_sections,
        key="analytics_section"
    )
    
    # Refresh data button
    if st.sidebar.button("🔄 Refresh Analytics", help="Reload all analytics data"):
        st.rerun()
    
    # Main content area
    if selected_section == "📈 Overview Dashboard":
        show_overview_dashboard(analytics)
    elif selected_section == "📚 Document Analytics":
        show_document_analytics(analytics)
    elif selected_section == "🕸️ Entity Network Analysis":
        show_entity_analysis(analytics)
    elif selected_section == "📋 Collection Insights":
        show_collection_insights(analytics)
    elif selected_section == "🔍 Knowledge Gap Analysis":
        show_knowledge_gaps(analytics)
    elif selected_section == "💡 Recommendations":
        show_recommendations(analytics)

def show_overview_dashboard(analytics: KnowledgeAnalytics):
    """Display high-level overview metrics"""
    st.header("📈 Knowledge Base Overview")
    
    # Get comprehensive stats
    doc_stats = analytics.get_document_stats()
    entity_analysis = analytics.get_entity_analysis()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Documents",
            value=doc_stats['total_documents'],
            delta=None
        )
    
    with col2:
        st.metric(
            label="Total Entities",
            value=doc_stats['total_entities'],
            delta=None
        )
    
    with col3:
        st.metric(
            label="Relationships",
            value=doc_stats['total_relationships'],
            delta=None
        )
    
    with col4:
        st.metric(
            label="Collections",
            value=doc_stats['collection_stats']['total_collections'],
            delta=None
        )
    
    st.divider()
    
    # Quick insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Document Type Distribution")
        if doc_stats['document_types']:
            df_doc_types = pd.DataFrame(list(doc_stats['document_types'].items()), 
                                      columns=['Type', 'Count'])
            fig = px.pie(df_doc_types, values='Count', names='Type', 
                        title="Documents by Type")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No document type data available")
    
    with col2:
        st.subheader("🕸️ Entity Type Distribution")
        if entity_analysis['entity_types']:
            df_entity_types = pd.DataFrame(list(entity_analysis['entity_types'].items()), 
                                         columns=['Type', 'Count'])
            fig = px.bar(df_entity_types, x='Type', y='Count', 
                        title="Entities by Type")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No entity data available")

def show_document_analytics(analytics: KnowledgeAnalytics):
    """Display detailed document analytics"""
    st.header("📚 Document Analytics")
    
    doc_stats = analytics.get_document_stats()
    
    # Show basic document statistics first
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", doc_stats['total_documents'])
    with col2:
        st.metric("Total Entities", doc_stats['total_entities'])
    with col3:
        st.metric("Total Relationships", doc_stats['total_relationships'])
    
    st.divider()
    
    # Document ingestion timeline
    if doc_stats['ingestion_timeline']:
        st.subheader("📅 Ingestion Timeline")
        
        try:
            df_timeline = pd.DataFrame(doc_stats['ingestion_timeline'])
            df_timeline['timestamp'] = pd.to_datetime(df_timeline['timestamp'])
            df_timeline['date'] = df_timeline['timestamp'].dt.date
            
            # Group by date and sum documents
            daily_stats = df_timeline.groupby('date')['documents_processed'].sum().reset_index()
            
            fig = px.line(daily_stats, x='date', y='documents_processed',
                         title="Documents Processed Over Time",
                         labels={'documents_processed': 'Documents', 'date': 'Date'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Show timeline data table
            st.subheader("Recent Ingestion Sessions")
            st.dataframe(df_timeline[['timestamp', 'documents_processed']].sort_values('timestamp', ascending=False).head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying timeline: {e}")
            st.write("Raw timeline data:", doc_stats['ingestion_timeline'])
    else:
        st.info("📊 No ingestion timeline data available. Timeline will appear after running document ingestion.")
    
    # Document type analysis
    if doc_stats['document_types']:
        st.subheader("📄 Document Type Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Document count by type
            df_types = pd.DataFrame(list(doc_stats['document_types'].items()), 
                                  columns=['Document Type', 'Count'])
            fig = px.bar(df_types, x='Document Type', y='Count',
                        title="Document Count by Type")
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Document type statistics table
            df_types['Percentage'] = (df_types['Count'] / df_types['Count'].sum() * 100).round(1)
            st.dataframe(df_types, use_container_width=True)
    else:
        st.info("📊 No document type data available. This will appear after ingesting documents into the knowledge base.")

def show_entity_analysis(analytics: KnowledgeAnalytics):
    """Display entity network analysis"""
    st.header("🕸️ Entity Network Analysis")
    
    entity_analysis = analytics.get_entity_analysis()
    
    # Entity type overview
    if entity_analysis['entity_types']:
        st.subheader("👥 Entity Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            df_entities = pd.DataFrame(list(entity_analysis['entity_types'].items()), 
                                     columns=['Entity Type', 'Count'])
            fig = px.pie(df_entities, values='Count', names='Entity Type',
                        title="Entity Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            total_entities = sum(entity_analysis['entity_types'].values())
            st.metric("Total Entities", total_entities)
            
            # Show entity breakdown
            for entity_type, count in entity_analysis['entity_types'].items():
                percentage = (count / total_entities * 100) if total_entities > 0 else 0
                st.write(f"**{entity_type.title()}**: {count} ({percentage:.1f}%)")
    
    # Top connected entities
    if entity_analysis['top_entities']:
        st.subheader("🌟 Most Connected Entities")
        
        for entity_type, entities in entity_analysis['top_entities'].items():
            if entities:
                st.write(f"**Top {entity_type.title()} Entities:**")
                
                df_top = pd.DataFrame(entities)
                fig = px.bar(df_top, x='name', y='connections',
                           title=f"Top {entity_type.title()} by Connections",
                           labels={'connections': 'Number of Connections', 'name': 'Entity'})
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    # Relationship patterns
    if entity_analysis['relationship_patterns']:
        st.subheader("🔗 Relationship Patterns")
        
        df_relationships = pd.DataFrame(list(entity_analysis['relationship_patterns'].items()), 
                                      columns=['Relationship Type', 'Count'])
        df_relationships = df_relationships.sort_values('Count', ascending=False)
        
        fig = px.bar(df_relationships, x='Relationship Type', y='Count',
                    title="Relationship Types in Knowledge Graph")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Entity clusters
    if entity_analysis['entity_clusters']:
        st.subheader("🎯 Entity Clusters")
        st.write("Groups of closely connected entities:")
        
        for cluster in entity_analysis['entity_clusters'][:5]:  # Show top 5 clusters
            with st.expander(f"Cluster {cluster['cluster_id']} - {cluster['size']} entities"):
                for entity in cluster['entities']:
                    st.write(f"- **{entity['name']}** ({entity['type']}) - {entity['connections']} connections")

def show_collection_insights(analytics: KnowledgeAnalytics):
    """Display collection analytics and insights"""
    st.header("📋 Collection Insights")
    
    doc_stats = analytics.get_document_stats()
    collection_stats = doc_stats['collection_stats']
    
    if not collection_stats['collection_sizes']:
        st.info("No collections found. Create collections in Collection Management to see insights.")
        return
    
    # Collection overview
    st.subheader("📊 Collection Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Collection sizes
        df_collections = pd.DataFrame(list(collection_stats['collection_sizes'].items()), 
                                    columns=['Collection', 'Document Count'])
        fig = px.bar(df_collections, x='Collection', y='Document Count',
                    title="Documents per Collection")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Collection statistics
        total_docs_in_collections = sum(collection_stats['collection_sizes'].values())
        avg_collection_size = total_docs_in_collections / len(collection_stats['collection_sizes'])
        
        st.metric("Total Collections", len(collection_stats['collection_sizes']))
        st.metric("Average Collection Size", f"{avg_collection_size:.1f}")
        st.metric("Total Docs in Collections", total_docs_in_collections)
    
    # Collection overlap analysis
    if collection_stats['collection_overlap']:
        st.subheader("🔄 Collection Overlap Analysis")
        
        df_overlap = pd.DataFrame(list(collection_stats['collection_overlap'].items()), 
                                columns=['Collection Pair', 'Overlap %'])
        df_overlap = df_overlap.sort_values('Overlap %', ascending=False)
        
        fig = px.bar(df_overlap, x='Collection Pair', y='Overlap %',
                    title="Collection Overlap Percentages")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations based on overlap
        high_overlap = df_overlap[df_overlap['Overlap %'] > 70]
        if not high_overlap.empty:
            st.warning("⚠️ High overlap detected between some collections. Consider consolidating:")
            for _, row in high_overlap.iterrows():
                st.write(f"- {row['Collection Pair']}: {row['Overlap %']:.1f}% overlap")

def show_knowledge_gaps(analytics: KnowledgeAnalytics):
    """Display knowledge gap analysis"""
    st.header("🔍 Knowledge Gap Analysis")
    
    gaps = analytics.identify_knowledge_gaps()
    
    # Isolated entities
    if gaps['isolated_entities']:
        st.subheader("🏝️ Isolated Entities")
        st.write("Entities with few connections that might benefit from additional documentation:")
        
        df_isolated = pd.DataFrame(gaps['isolated_entities'])
        st.dataframe(df_isolated, use_container_width=True)
    
    # Sparse document types
    if gaps['sparse_document_types']:
        st.subheader("📄 Underrepresented Document Types")
        st.write("Document types with low representation in the knowledge base:")
        
        df_sparse = pd.DataFrame(gaps['sparse_document_types'])
        fig = px.bar(df_sparse, x='type', y='percentage',
                    title="Document Type Representation",
                    labels={'percentage': 'Percentage of Total Documents', 'type': 'Document Type'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_sparse, use_container_width=True)
    
    # Knowledge graph density analysis
    if analytics.graph_manager and analytics.graph_manager.graph:
        st.subheader("🕸️ Knowledge Graph Density")
        graph = analytics.graph_manager.graph
        
        # Calculate graph metrics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        density = (num_edges / max_possible_edges) * 100 if max_possible_edges > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Graph Density", f"{density:.2f}%")
        with col2:
            st.metric("Average Connections", f"{(2 * num_edges / num_nodes):.1f}" if num_nodes > 0 else "0")
        with col3:
            # Handle directed vs undirected graphs
            if graph.is_directed():
                num_components = nx.number_weakly_connected_components(graph)
                st.metric("Weakly Connected Components", str(num_components))
            else:
                num_components = nx.number_connected_components(graph)
                st.metric("Connected Components", str(num_components))
        
        # Density interpretation
        if density < 1:
            st.info("🔍 Low graph density suggests opportunities for discovering more relationships between entities.")
        elif density > 10:
            st.success("✅ High graph density indicates a well-connected knowledge base.")
        else:
            st.info("📊 Moderate graph density - good foundation with room for growth.")

def show_recommendations(analytics: KnowledgeAnalytics):
    """Display actionable recommendations"""
    st.header("💡 Recommendations")
    
    # Get all analysis data for comprehensive recommendations
    doc_stats = analytics.get_document_stats()
    entity_analysis = analytics.get_entity_analysis()
    gaps = analytics.identify_knowledge_gaps()
    
    # Prioritized recommendations
    st.subheader("🎯 Priority Recommendations")
    
    recommendations = []
    
    # Document diversity recommendations
    if doc_stats['document_types']:
        doc_counts = list(doc_stats['document_types'].values())
        if max(doc_counts) > sum(doc_counts) * 0.7:  # One type dominates
            recommendations.append({
                'priority': 'High',
                'category': 'Document Diversity',
                'recommendation': 'Diversify document types to improve knowledge coverage balance.',
                'action': 'Add more variety in document types during next ingestion.'
            })
    
    # Entity connection recommendations
    if gaps['isolated_entities']:
        recommendations.append({
            'priority': 'Medium',
            'category': 'Entity Connectivity',
            'recommendation': f'Strengthen connections for {len(gaps["isolated_entities"])} weakly connected entities.',
            'action': 'Add documents that reference these entities alongside others.'
        })
    
    # Collection optimization
    collection_overlap = doc_stats['collection_stats'].get('collection_overlap', {})
    high_overlap_collections = [k for k, v in collection_overlap.items() if v > 80]
    if high_overlap_collections:
        recommendations.append({
            'priority': 'Low',
            'category': 'Collection Management',
            'recommendation': 'Consider consolidating collections with high overlap.',
            'action': f'Review and potentially merge: {", ".join(high_overlap_collections[:3])}'
        })
    
    # Display recommendations
    for rec in recommendations:
        priority_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
        with st.expander(f"{priority_color[rec['priority']]} {rec['category']} - {rec['priority']} Priority"):
            st.write(f"**Recommendation:** {rec['recommendation']}")
            st.write(f"**Suggested Action:** {rec['action']}")
    
    # Performance insights
    st.subheader("📈 Performance Insights")
    
    if doc_stats['total_documents'] > 0:
        entity_to_doc_ratio = doc_stats['total_entities'] / doc_stats['total_documents']
        rel_to_entity_ratio = doc_stats['total_relationships'] / doc_stats['total_entities'] if doc_stats['total_entities'] > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Entities per Document", f"{entity_to_doc_ratio:.1f}")
            if entity_to_doc_ratio < 5:
                st.info("💡 Consider documents with richer entity content")
            elif entity_to_doc_ratio > 15:
                st.success("✅ Good entity extraction rate")
        
        with col2:
            st.metric("Relationships per Entity", f"{rel_to_entity_ratio:.1f}")
            if rel_to_entity_ratio < 1.5:
                st.info("💡 Opportunity to discover more entity relationships")
            elif rel_to_entity_ratio > 3:
                st.success("✅ Rich relationship network")

if __name__ == "__main__":
    main()
