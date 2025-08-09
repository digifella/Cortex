#!/usr/bin/env python3
# ## File: examples/api_usage_examples.py
# Version: 1.0.0
# Date: 2025-07-27
# Purpose: Comprehensive examples demonstrating Cortex Suite API usage.

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import requests
from datetime import datetime

class CortexAPIClient:
    """Python client for Cortex Suite API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", token: str = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        if token:
            self.session.headers.update({'Authorization': f'Bearer {token}'})
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = self.session.get(f'{self.base_url}/health')
        response.raise_for_status()
        return response.json()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        response = self.session.get(f'{self.base_url}/status')
        response.raise_for_status()
        return response.json()
    
    def search(self, 
               query: str, 
               search_type: str = 'hybrid',
               max_results: int = 10,
               include_synthesis: bool = True,
               similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Search the knowledge base"""
        payload = {
            'query': query,
            'search_type': search_type,
            'max_results': max_results,
            'include_synthesis': include_synthesis,
            'similarity_threshold': similarity_threshold
        }
        
        response = self.session.post(f'{self.base_url}/search', json=payload)
        response.raise_for_status()
        return response.json()
    
    def batch_search(self,
                    queries: List[str],
                    search_type: str = 'hybrid',
                    include_synthesis: bool = True) -> List[Dict[str, Any]]:
        """Perform multiple searches concurrently"""
        payload = {
            'queries': queries,
            'search_type': search_type,
            'include_synthesis': include_synthesis
        }
        
        response = self.session.post(f'{self.base_url}/search/batch', json=payload)
        response.raise_for_status()
        return response.json()
    
    def ingest_documents(self,
                        file_paths: List[str],
                        exclusion_patterns: List[str] = None,
                        max_concurrent: int = 5) -> Dict[str, Any]:
        """Ingest documents into the knowledge base"""
        payload = {
            'file_paths': file_paths,
            'exclusion_patterns': exclusion_patterns or [],
            'max_concurrent': max_concurrent
        }
        
        response = self.session.post(f'{self.base_url}/ingest', json=payload)
        response.raise_for_status()
        return response.json()
    
    def upload_file(self, file_path: str) -> Dict[str, Any]:
        """Upload and ingest a single file"""
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/octet-stream')}
            # Temporarily remove Content-Type header for file upload
            headers = {k: v for k, v in self.session.headers.items() if k != 'Content-Type'}
            response = self.session.post(f'{self.base_url}/ingest/upload', 
                                       files=files, headers=headers)
        response.raise_for_status()
        return response.json()
    
    def get_collections(self) -> List[Dict[str, Any]]:
        """Get all document collections"""
        response = self.session.get(f'{self.base_url}/collections')
        response.raise_for_status()
        return response.json()
    
    def create_collection(self,
                         name: str,
                         description: str,
                         documents: List[str] = None) -> Dict[str, Any]:
        """Create a new collection"""
        data = {
            'description': description,
            'documents': documents or []
        }
        
        response = self.session.post(f'{self.base_url}/collections/{name}', data=data)
        response.raise_for_status()
        return response.json()
    
    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """Get information about a specific entity"""
        response = self.session.get(f'{self.base_url}/entities/{entity_name}')
        response.raise_for_status()
        return response.json()
    
    def create_backup(self,
                     backup_path: str,
                     include_images: bool = True,
                     compress: bool = True) -> Dict[str, Any]:
        """Create a backup of the knowledge base"""
        payload = {
            'backup_path': backup_path,
            'include_images': include_images,
            'compress': compress
        }
        
        response = self.session.post(f'{self.base_url}/backup/create', json=payload)
        response.raise_for_status()
        return response.json()
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        response = self.session.get(f'{self.base_url}/backup/list')
        response.raise_for_status()
        return response.json()
    
    def restore_backup(self,
                      backup_id: str,
                      overwrite_existing: bool = False,
                      verify_checksum: bool = True) -> Dict[str, Any]:
        """Restore from a backup"""
        payload = {
            'backup_id': backup_id,
            'overwrite_existing': overwrite_existing,
            'verify_checksum': verify_checksum
        }
        
        response = self.session.post(f'{self.base_url}/backup/restore', json=payload)
        response.raise_for_status()
        return response.json()
    
    def delete_backup(self, backup_id: str) -> Dict[str, Any]:
        """Delete a specific backup"""
        response = self.session.delete(f'{self.base_url}/backup/{backup_id}')
        response.raise_for_status()
        return response.json()
    
    def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """Verify backup integrity"""
        response = self.session.get(f'{self.base_url}/backup/{backup_id}/verify')
        response.raise_for_status()
        return response.json()

def example_basic_operations():
    """Example: Basic API operations"""
    print("="*60)
    print("🔧 Basic Operations Example")
    print("="*60)
    
    # Initialize client
    client = CortexAPIClient()
    
    try:
        # Health check
        print("\n1. Health Check")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Version: {health['version']}")
        
        # System status
        print("\n2. System Status")
        status = client.get_system_status()
        print(f"   Documents: {status['total_documents']}")
        print(f"   Entities: {status['total_entities']}")
        print(f"   Collections: {status['collections_count']}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")

def example_search_operations():
    """Example: Search operations"""
    print("="*60)
    print("🔍 Search Operations Example")
    print("="*60)
    
    client = CortexAPIClient()
    
    try:
        # Single search
        print("\n1. Single Search")
        query = "machine learning artificial intelligence"
        results = client.search(query, search_type='hybrid')
        
        print(f"   Query: {query}")
        print(f"   Results: {results['total_results']}")
        print(f"   Processing time: {results['processing_time']:.2f}s")
        
        if results['synthesis']:
            print(f"   Synthesis: {results['synthesis'][:100]}...")
        
        # Show top results
        for i, result in enumerate(results['results'][:3]):
            print(f"   Result {i+1}: {result['metadata'].get('file_name', 'Unknown')} "
                  f"(similarity: {result.get('similarity', 0):.2f})")
        
        # Batch search
        print("\n2. Batch Search")
        queries = [
            "neural networks deep learning",
            "natural language processing",
            "computer vision image recognition"
        ]
        
        batch_results = client.batch_search(queries, search_type='vector')
        
        for i, result in enumerate(batch_results):
            print(f"   Query {i+1}: '{result['query']}' -> {result['total_results']} results")
        
        # Search with different parameters
        print("\n3. Vector Search with High Threshold")
        vector_results = client.search(
            "research methodology",
            search_type='vector',
            max_results=5,
            similarity_threshold=0.9
        )
        print(f"   High threshold results: {vector_results['total_results']}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Search error: {e}")

def example_ingestion_operations():
    """Example: Document ingestion operations"""
    print("="*60)
    print("📄 Ingestion Operations Example")
    print("="*60)
    
    client = CortexAPIClient()
    
    # Create sample documents for demonstration
    sample_docs = create_sample_documents()
    
    try:
        # Batch ingestion
        print("\n1. Batch Document Ingestion")
        file_paths = list(sample_docs.values())
        
        ingestion_result = client.ingest_documents(
            file_paths=file_paths,
            exclusion_patterns=["*.tmp", "*~*"],
            max_concurrent=3
        )
        
        print(f"   Processed: {ingestion_result['success_count']} documents")
        print(f"   Errors: {ingestion_result['error_count']}")
        print(f"   Entities extracted: {ingestion_result['total_entities']}")
        print(f"   Relationships: {ingestion_result['total_relationships']}")
        print(f"   Processing time: {ingestion_result['processing_time']:.2f}s")
        
        # File upload
        print("\n2. File Upload")
        if file_paths:
            upload_result = client.upload_file(file_paths[0])
            print(f"   Uploaded: {upload_result['filename']}")
            print(f"   Status: {upload_result['status']}")
            print(f"   Entities: {upload_result['entities_extracted']}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Ingestion error: {e}")
    finally:
        # Clean up sample documents
        cleanup_sample_documents(sample_docs)

def example_collection_management():
    """Example: Collection management operations"""
    print("="*60)
    print("📚 Collection Management Example")
    print("="*60)
    
    client = CortexAPIClient()
    
    try:
        # List existing collections
        print("\n1. List Existing Collections")
        collections = client.get_collections()
        print(f"   Found {len(collections)} collections:")
        
        for collection in collections[:3]:
            print(f"   - {collection['name']}: {len(collection['documents'])} documents")
        
        # Create new collection
        print("\n2. Create New Collection")
        collection_name = f"example_collection_{int(time.time())}"
        
        create_result = client.create_collection(
            name=collection_name,
            description="Example collection created via API",
            documents=["example1.pdf", "example2.docx"]
        )
        print(f"   Created: {create_result['message']}")
        
        # List collections again to verify
        updated_collections = client.get_collections()
        print(f"   Total collections now: {len(updated_collections)}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Collection error: {e}")

def example_entity_operations():
    """Example: Entity management operations"""
    print("="*60)
    print("👤 Entity Operations Example")
    print("="*60)
    
    client = CortexAPIClient()
    
    # Sample entity names to look up
    sample_entities = [
        "Project Cortex",
        "John Smith", 
        "Machine Learning",
        "Artificial Intelligence"
    ]
    
    try:
        print("\n1. Entity Information Lookup")
        
        for entity_name in sample_entities:
            try:
                entity_info = client.get_entity_info(entity_name)
                print(f"   ✅ {entity_name}:")
                print(f"      Type: {entity_info['type']}")
                print(f"      Relationships: {len(entity_info['relationships'])}")
                
                # Show some properties
                if entity_info.get('properties'):
                    props = list(entity_info['properties'].keys())[:3]
                    print(f"      Properties: {', '.join(props)}")
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"   ❌ {entity_name}: Not found")
                else:
                    print(f"   ❌ {entity_name}: Error {e.response.status_code}")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Entity error: {e}")

def example_backup_operations():
    """Example: Backup and restore operations"""
    print("="*60)
    print("💾 Backup Operations Example")
    print("="*60)
    
    client = CortexAPIClient()
    
    try:
        # List existing backups
        print("\n1. List Existing Backups")
        backups = client.list_backups()
        print(f"   Found {len(backups)} backups:")
        
        for backup in backups[:3]:
            size_mb = backup['total_size'] / (1024 * 1024)
            print(f"   - {backup['backup_id']}: {backup['file_count']} files, "
                  f"{size_mb:.1f}MB, {backup['compression']}")
        
        # Create new backup
        print("\n2. Create New Backup")
        backup_path = f"/tmp/cortex_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_result = client.create_backup(
            backup_path=backup_path,
            include_images=True,
            compress=True
        )
        
        print(f"   Created backup: {backup_result['backup_id']}")
        print(f"   Files: {backup_result['file_count']}")
        print(f"   Size: {backup_result['total_size'] / (1024*1024):.1f}MB")
        print(f"   Compression: {backup_result['compression']}")
        
        # Verify backup integrity
        print("\n3. Verify Backup Integrity")
        verification = client.verify_backup(backup_result['backup_id'])
        print(f"   Verification: {'✅ Valid' if verification['valid'] else '❌ Invalid'}")
        
        # Optional: Clean up the test backup
        print(f"\n4. Cleanup Test Backup")
        delete_result = client.delete_backup(backup_result['backup_id'])
        print(f"   Cleanup: {delete_result['message']}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Backup error: {e}")

def example_performance_monitoring():
    """Example: Performance monitoring and analysis"""
    print("="*60)
    print("📊 Performance Monitoring Example")
    print("="*60)
    
    client = CortexAPIClient()
    
    try:
        # Baseline system status
        print("\n1. Baseline Metrics")
        baseline_status = client.get_system_status()
        baseline_time = time.time()
        
        print(f"   Documents: {baseline_status['total_documents']}")
        print(f"   Entities: {baseline_status['total_entities']}")
        
        # Performance test: Multiple searches
        print("\n2. Performance Test - Batch Searches")
        test_queries = [
            "machine learning algorithms",
            "artificial intelligence research", 
            "neural network architectures",
            "deep learning applications",
            "natural language processing"
        ]
        
        start_time = time.time()
        batch_results = client.batch_search(test_queries, search_type='hybrid')
        end_time = time.time()
        
        total_results = sum(result['total_results'] for result in batch_results)
        avg_processing_time = sum(result['processing_time'] for result in batch_results) / len(batch_results)
        
        print(f"   Queries: {len(test_queries)}")
        print(f"   Total results: {total_results}")
        print(f"   Wall clock time: {end_time - start_time:.2f}s")
        print(f"   Avg processing time: {avg_processing_time:.2f}s")
        print(f"   Queries per second: {len(test_queries) / (end_time - start_time):.2f}")
        
        # Individual search performance
        print("\n3. Individual Search Performance")
        performance_results = []
        
        for query in test_queries[:3]:  # Test first 3 queries individually
            start = time.time()
            result = client.search(query, search_type='vector', include_synthesis=False)
            end = time.time()
            
            performance_results.append({
                'query': query,
                'results': result['total_results'],
                'wall_time': end - start,
                'processing_time': result['processing_time']
            })
        
        for perf in performance_results:
            print(f"   '{perf['query'][:30]}...' -> {perf['results']} results, "
                  f"{perf['wall_time']:.2f}s wall, {perf['processing_time']:.2f}s processing")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Performance monitoring error: {e}")

def example_error_handling():
    """Example: Error handling patterns"""
    print("="*60)
    print("⚠️ Error Handling Example")
    print("="*60)
    
    client = CortexAPIClient()
    
    # Test various error conditions
    error_scenarios = [
        {
            'name': 'Invalid search type',
            'action': lambda: client.search("test", search_type='invalid_type')
        },
        {
            'name': 'Non-existent entity',
            'action': lambda: client.get_entity_info("NonExistentEntity12345")
        },
        {
            'name': 'Invalid file paths',
            'action': lambda: client.ingest_documents(["/non/existent/file.txt"])
        },
        {
            'name': 'Non-existent backup',
            'action': lambda: client.delete_backup("non_existent_backup")
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\n{scenario['name']}:")
        
        try:
            result = scenario['action']()
            print(f"   ✅ Unexpected success: {result}")
        
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json().get('detail', 'No detail provided')
            except:
                error_detail = e.response.text
            
            print(f"   ❌ HTTP {status_code}: {error_detail}")
        
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed: {str(e)}")
        
        except Exception as e:
            print(f"   ❌ Unexpected error: {str(e)}")

def create_sample_documents() -> Dict[str, str]:
    """Create sample documents for testing"""
    import tempfile
    
    sample_content = {
        'doc1.txt': """
        This is a sample document about machine learning and artificial intelligence.
        It discusses various algorithms including neural networks, decision trees, and support vector machines.
        The document covers both supervised and unsupervised learning techniques.
        """,
        'doc2.txt': """
        Research paper on natural language processing and computational linguistics.
        Authors: Dr. Jane Smith and Prof. John Doe from MIT.
        This work explores transformer architectures and attention mechanisms.
        """,
        'doc3.txt': """
        Technical documentation for the Cortex Suite knowledge management system.
        The system uses vector databases and graph structures for information retrieval.
        Key components include entity extraction, relationship mapping, and hybrid search.
        """
    }
    
    temp_files = {}
    temp_dir = Path(tempfile.mkdtemp(prefix="cortex_api_example_"))
    
    for filename, content in sample_content.items():
        file_path = temp_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        temp_files[filename] = str(file_path)
    
    return temp_files

def cleanup_sample_documents(sample_docs: Dict[str, str]):
    """Clean up sample documents"""
    import shutil
    
    if sample_docs:
        # Get the parent directory of first file
        first_file = Path(list(sample_docs.values())[0])
        temp_dir = first_file.parent
        
        if temp_dir.exists() and "cortex_api_example_" in str(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """Run all examples"""
    print("🚀 Cortex Suite API Usage Examples")
    print("="*60)
    print("This script demonstrates various API operations.")
    print("Make sure the API server is running at http://localhost:8000")
    print()
    
    examples = [
        ("Basic Operations", example_basic_operations),
        ("Search Operations", example_search_operations),
        ("Ingestion Operations", example_ingestion_operations),
        ("Collection Management", example_collection_management),
        ("Entity Operations", example_entity_operations),
        ("Backup Operations", example_backup_operations),
        ("Performance Monitoring", example_performance_monitoring),
        ("Error Handling", example_error_handling)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            example_func()
            print(f"✅ {name} completed successfully")
        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted during {name}")
            break
        except Exception as e:
            print(f"❌ {name} failed: {e}")
        
        # Brief pause between examples
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print("🎉 All examples completed!")
    print("Check the API documentation at http://localhost:8000/docs for more details.")

if __name__ == "__main__":
    main()