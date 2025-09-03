#!/usr/bin/env python3
"""
GraphRAG Retroactive Extraction Utility
Purpose: Add GraphRAG entity extraction to existing knowledge bases without full re-ingestion
Date: 2025-09-01
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import chromadb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.utils import get_logger, convert_windows_to_wsl_path
from cortex_engine.entity_extractor import EntityExtractor
from cortex_engine.graph_manager import EnhancedGraphManager
from cortex_engine.config_manager import ConfigManager

logger = get_logger(__name__)

class GraphRAGRetroactiveExtractor:
    """Utility to add GraphRAG entities to existing knowledge bases"""
    
    def __init__(self, db_path: str):
        self.db_path = convert_to_docker_mount_path(db_path)
        self.chroma_db_path = os.path.join(self.db_path, "knowledge_hub_db")
        self.graph_file_path = os.path.join(self.db_path, "knowledge_cortex.gpickle")
        
        # Initialize components
        self.entity_extractor = EntityExtractor()
        self.graph_manager = EnhancedGraphManager(self.graph_file_path)
        
        # ChromaDB connection
        self.chroma_client = None
        self.collection = None
        
    def connect_to_chromadb(self):
        """Connect to existing ChromaDB collection"""
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
            collections = self.chroma_client.list_collections()
            
            if not collections:
                raise ValueError(f"No ChromaDB collections found in {self.chroma_db_path}")
            
            # Use the first collection (or look for specific name)
            collection_name = collections[0].name
            self.collection = self.chroma_client.get_collection(collection_name)
            
            logger.info(f"Connected to ChromaDB collection '{collection_name}' with {self.collection.count()} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            return False
    
    def extract_entities_from_existing_documents(self, batch_size: int = 50) -> Dict:
        """Extract entities from all existing documents in ChromaDB"""
        if not self.connect_to_chromadb():
            return {"success": False, "error": "Could not connect to ChromaDB"}
        
        try:
            # Get all documents
            total_docs = self.collection.count()
            logger.info(f"Processing {total_docs} existing documents for entity extraction...")
            
            processed = 0
            entities_added = 0
            relationships_added = 0
            
            # Process documents in batches
            offset = 0
            while offset < total_docs:
                # Get batch of documents
                batch_results = self.collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=['documents', 'metadatas']
                )
                
                for i, (doc_text, metadata) in enumerate(zip(
                    batch_results['documents'], 
                    batch_results['metadatas']
                )):
                    try:
                        # Extract entities and relationships
                        entities, relationships = self.entity_extractor.extract_entities_and_relationships(
                            doc_text, metadata or {}
                        )
                        
                        # Add entities to graph
                        for entity in entities:
                            entity_id = f"{entity.entity_type}:{entity.name}"
                            self.graph_manager.add_entity(
                                entity_id=entity_id,
                                entity_type=entity.entity_type,
                                confidence=entity.confidence,
                                aliases=entity.aliases,
                                extraction_method=entity.extraction_method
                            )
                            entities_added += 1
                        
                        # Add relationships to graph
                        for rel in relationships:
                            source_id = f"{rel.source_entity_type}:{rel.source_entity}"
                            target_id = f"{rel.target_entity_type}:{rel.target_entity}"
                            
                            self.graph_manager.add_relationship(
                                source_id=source_id,
                                target_id=target_id,
                                relationship_type=rel.relationship_type,
                                confidence=rel.confidence,
                                context=rel.context
                            )
                            relationships_added += 1
                        
                        processed += 1
                        
                        if processed % 10 == 0:
                            logger.info(f"Processed {processed}/{total_docs} documents...")
                            
                    except Exception as e:
                        logger.warning(f"Failed to process document {offset + i}: {e}")
                        continue
                
                offset += batch_size
            
            # Save the updated graph
            self.graph_manager.save_graph()
            
            return {
                "success": True,
                "documents_processed": processed,
                "entities_added": entities_added,
                "relationships_added": relationships_added,
                "total_entities": self.graph_manager.graph.number_of_nodes(),
                "total_relationships": self.graph_manager.graph.number_of_edges()
            }
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"success": False, "error": str(e)}

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add GraphRAG entities to existing knowledge base")
    parser.add_argument("--db-path", help="Path to knowledge base database")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Get database path
    db_path = args.db_path
    if not db_path:
        # Try to get from config
        config_manager = ConfigManager()
        config = config_manager._load_config()
        db_path = config.get('ai_database_path')
        
        if not db_path:
            print("❌ No database path provided. Use --db-path or configure in settings.")
            sys.exit(1)
    
    print(f"🚀 Starting GraphRAG retroactive extraction...")
    print(f"📁 Database path: {db_path}")
    
    # Run extraction
    extractor = GraphRAGRetroactiveExtractor(db_path)
    result = extractor.extract_entities_from_existing_documents(args.batch_size)
    
    if result["success"]:
        print(f"✅ Successfully completed GraphRAG extraction!")
        print(f"📊 Documents processed: {result['documents_processed']}")
        print(f"🏷️  Entities added: {result['entities_added']}")
        print(f"🔗 Relationships added: {result['relationships_added']}")
        print(f"📈 Total entities in graph: {result['total_entities']}")
        print(f"📈 Total relationships in graph: {result['total_relationships']}")
    else:
        print(f"❌ GraphRAG extraction failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
