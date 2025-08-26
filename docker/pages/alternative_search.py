"""
Alternative search approach bypassing LlamaIndex query engine
This is a fallback method if the main search continues to fail
"""

import chromadb
import streamlit as st
from chromadb.config import Settings as ChromaSettings
from cortex_engine.utils import convert_windows_to_wsl_path, get_logger
from cortex_engine.config import COLLECTION_NAME
import os

logger = get_logger(__name__)

def direct_chromadb_search(db_path, query, top_k=20):
    """
    Perform direct ChromaDB search bypassing LlamaIndex entirely
    """
    try:
        wsl_db_path = convert_windows_to_wsl_path(db_path)
        chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
        
        if not os.path.isdir(chroma_db_path):
            st.error(f"Database not found: {chroma_db_path}")
            return []
        
        # Direct ChromaDB client
        db_settings = ChromaSettings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
        collection = client.get_collection(COLLECTION_NAME)
        
        # Direct query without any where clauses
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results and results.get('documents'):
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                formatted_results.append({
                    'rank': i + 1,
                    'score': 1.0 - distance,  # Convert distance to similarity score
                    'text': doc,
                    'file_path': metadata.get('file_path', 'Unknown'),
                    'file_name': metadata.get('file_name', 'Unknown'),
                    'document_type': metadata.get('document_type', 'Unknown'),
                    'chunk_id': metadata.get('chunk_id', f'chunk_{i}'),
                    'doc_id': metadata.get('doc_id', f'doc_{i}')
                })
        
        logger.info(f"Direct ChromaDB search returned {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Direct ChromaDB search failed: {e}")
        st.error(f"Direct search failed: {e}")
        return []

def test_direct_search():
    """Test function for direct search"""
    st.subheader("🧪 Direct ChromaDB Search Test")
    
    # Get database path from session state or config
    from cortex_engine.config_manager import ConfigManager
    config_manager = ConfigManager()
    current_config = config_manager.get_config()
    db_path = current_config.get("ai_database_path", "")
    
    if not db_path:
        st.error("Database path not configured")
        return
    
    test_query = st.text_input("Test query:", value="pedagogy")
    
    if st.button("🔍 Test Direct Search"):
        if test_query:
            results = direct_chromadb_search(db_path, test_query)
            if results:
                st.success(f"Found {len(results)} results!")
                for result in results[:5]:  # Show first 5 results
                    with st.expander(f"Result {result['rank']}: {result['file_name']}"):
                        st.write(f"**Score:** {result['score']:.3f}")
                        st.write(f"**Type:** {result['document_type']}")
                        st.write(f"**Text:** {result['text'][:200]}...")
            else:
                st.warning("No results found")
        else:
            st.warning("Please enter a test query")

if __name__ == "__main__":
    test_direct_search()