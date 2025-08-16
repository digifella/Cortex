# Cortex Suite REST API

A comprehensive REST API for the Cortex Suite AI-powered knowledge management and proposal generation system.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Authentication](#authentication)
4. [API Endpoints](#api-endpoints)
5. [Usage Examples](#usage-examples)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Development](#development)

## Overview

The Cortex Suite API provides programmatic access to:
- **Knowledge Base Search**: Vector, graph, and hybrid search capabilities
- **Document Ingestion**: Async document processing with entity extraction
- **Collection Management**: Organize and manage document collections
- **Backup & Restore**: Comprehensive data backup and recovery
- **Entity Management**: Access knowledge graph entities and relationships
- **System Monitoring**: Health checks and system statistics

### Key Features

- ⚡ **Async Processing**: High-performance concurrent operations
- 🔍 **Advanced Search**: Multiple search modes with AI synthesis
- 📚 **Knowledge Graph**: Entity extraction and relationship mapping
- 🔐 **Secure**: Token-based authentication and validation
- 📊 **Monitoring**: Built-in health checks and metrics
- 🗄️ **Backup**: Complete data backup and restore capabilities

## Getting Started

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the API server:
```bash
# Development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

3. Access API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Environment Variables

Create a `.env` file with required configuration:

```bash
# Database Configuration
AI_DATABASE_PATH="/path/to/your/database"
KNOWLEDGE_SOURCE_PATH="/path/to/your/documents"

# API Configuration
API_HOST="0.0.0.0"
API_PORT=8000
API_WORKERS=4

# Authentication (Optional)
API_SECRET_KEY="your-secret-key"
JWT_ALGORITHM="HS256"

# Logging
LOG_LEVEL="INFO"
```

## Authentication

The API uses Bearer token authentication. Include your token in the Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" \
     http://localhost:8000/search
```

**Note**: Authentication is currently implemented as a placeholder. Implement your specific authentication logic in the `verify_token` function.

## API Endpoints

### Health & Status

#### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-27T10:00:00Z",
  "version": "1.0.0"
}
```

#### GET /status
Get comprehensive system status.

**Response:**
```json
{
  "total_documents": 1250,
  "total_entities": 342,
  "total_relationships": 567,
  "database_size": 2048000,
  "collections_count": 15,
  "last_updated": "2025-07-27T10:00:00Z"
}
```

### Search Operations

#### POST /search
Search the knowledge base with various modes.

**Request Body:**
```json
{
  "query": "artificial intelligence machine learning",
  "search_type": "hybrid",
  "max_results": 10,
  "include_synthesis": true,
  "similarity_threshold": 0.7
}
```

**Response:**
```json
{
  "query": "artificial intelligence machine learning",
  "results": [
    {
      "content": "Document content excerpt...",
      "metadata": {
        "file_name": "ai_research.pdf",
        "doc_type": "Research Paper"
      },
      "similarity": 0.92,
      "rank": 1,
      "source": "vector"
    }
  ],
  "synthesis": "Based on the search results, artificial intelligence and machine learning...",
  "total_results": 15,
  "processing_time": 0.45,
  "status": "success"
}
```

#### POST /search/batch
Perform multiple searches concurrently.

**Request Body:**
```json
{
  "queries": [
    "machine learning algorithms",
    "neural networks deep learning",
    "natural language processing"
  ],
  "search_type": "hybrid",
  "include_synthesis": true
}
```

### Document Ingestion

#### POST /ingest
Ingest documents into the knowledge base.

**Request Body:**
```json
{
  "file_paths": [
    "/path/to/document1.pdf",
    "/path/to/document2.docx"
  ],
  "exclusion_patterns": ["*.tmp", "*~*"],
  "max_concurrent": 5
}
```

**Response:**
```json
{
  "success_count": 2,
  "error_count": 0,
  "skipped_count": 0,
  "processed_files": ["document1.pdf", "document2.docx"],
  "errors": [],
  "total_entities": 25,
  "total_relationships": 18,
  "processing_time": 12.34
}
```

#### POST /ingest/upload
Upload and ingest a single file.

**Request:**
```bash
curl -X POST \
  -F "file=@document.pdf" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/ingest/upload
```

### Collection Management

#### GET /collections
List all document collections.

**Response:**
```json
[
  {
    "name": "research_papers",
    "description": "Academic research papers",
    "documents": ["paper1.pdf", "paper2.pdf"],
    "created_date": "2025-07-27T10:00:00Z",
    "last_modified": "2025-07-27T12:00:00Z"
  }
]
```

#### POST /collections/{collection_name}
Create a new collection.

**Request:**
```bash
curl -X POST \
  -F "description=My new collection" \
  -F "documents=doc1.pdf" \
  -F "documents=doc2.pdf" \
  http://localhost:8000/collections/my_collection
```

### Entity Management

#### GET /entities/{entity_name}
Get information about a specific entity.

**Response:**
```json
{
  "name": "John Smith",
  "type": "person",
  "properties": {
    "role": "researcher",
    "affiliation": "University XYZ"
  },
  "relationships": ["collaborates_with", "authors", "works_on"]
}
```

### Backup & Restore

#### POST /backup/create
Create a backup of the knowledge base.

**Request Body:**
```json
{
  "backup_path": "/backups/cortex_backup_20250727",
  "include_images": true,
  "compress": true
}
```

#### GET /backup/list
List all available backups.

**Response:**
```json
[
  {
    "backup_id": "backup_20250727_100000",
    "backup_type": "full",
    "creation_time": "2025-07-27T10:00:00Z",
    "file_count": 1250,
    "total_size": 2048000,
    "compression": "gzip",
    "checksum": "a1b2c3d4e5f6..."
  }
]
```

#### POST /backup/restore
Restore from a backup.

**Request Body:**
```json
{
  "backup_id": "backup_20250727_100000",
  "overwrite_existing": false,
  "verify_checksum": true
}
```

#### DELETE /backup/{backup_id}
Delete a specific backup.

#### GET /backup/{backup_id}/verify
Verify backup integrity.

## Usage Examples

### Python Client Example

```python
import requests
import json

class CortexAPIClient:
    def __init__(self, base_url, token=None):
        self.base_url = base_url.rstrip('/')
        self.headers = {'Content-Type': 'application/json'}
        if token:
            self.headers['Authorization'] = f'Bearer {token}'
    
    def search(self, query, search_type='hybrid', max_results=10):
        """Search the knowledge base"""
        payload = {
            'query': query,
            'search_type': search_type,
            'max_results': max_results,
            'include_synthesis': True
        }
        
        response = requests.post(
            f'{self.base_url}/search',
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def ingest_documents(self, file_paths, max_concurrent=5):
        """Ingest documents into knowledge base"""
        payload = {
            'file_paths': file_paths,
            'max_concurrent': max_concurrent
        }
        
        response = requests.post(
            f'{self.base_url}/ingest',
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def create_backup(self, backup_path, compress=True):
        """Create a backup"""
        payload = {
            'backup_path': backup_path,
            'include_images': True,
            'compress': compress
        }
        
        response = requests.post(
            f'{self.base_url}/backup/create',
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

# Usage
client = CortexAPIClient('http://localhost:8000', 'your-token')

# Search example
results = client.search('machine learning algorithms')
print(f"Found {results['total_results']} results")
print(f"Synthesis: {results['synthesis']}")

# Ingestion example
ingestion_result = client.ingest_documents([
    '/path/to/document1.pdf',
    '/path/to/document2.docx'
])
print(f"Processed {ingestion_result['success_count']} documents")

# Backup example
backup_info = client.create_backup('/backups/my_backup')
print(f"Backup created: {backup_info['backup_id']}")
```

### JavaScript/Node.js Example

```javascript
class CortexAPIClient {
    constructor(baseUrl, token = null) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.headers = {
            'Content-Type': 'application/json'
        };
        if (token) {
            this.headers['Authorization'] = `Bearer ${token}`;
        }
    }

    async search(query, searchType = 'hybrid', maxResults = 10) {
        const payload = {
            query: query,
            search_type: searchType,
            max_results: maxResults,
            include_synthesis: true
        };

        const response = await fetch(`${this.baseUrl}/search`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Search failed: ${response.statusText}`);
        }

        return await response.json();
    }

    async batchSearch(queries, searchType = 'hybrid') {
        const payload = {
            queries: queries,
            search_type: searchType,
            include_synthesis: true
        };

        const response = await fetch(`${this.baseUrl}/search/batch`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Batch search failed: ${response.statusText}`);
        }

        return await response.json();
    }

    async getSystemStatus() {
        const response = await fetch(`${this.baseUrl}/status`, {
            headers: this.headers
        });

        if (!response.ok) {
            throw new Error(`Status check failed: ${response.statusText}`);
        }

        return await response.json();
    }
}

// Usage
const client = new CortexAPIClient('http://localhost:8000', 'your-token');

// Search example
client.search('artificial intelligence research')
    .then(results => {
        console.log(`Found ${results.total_results} results`);
        console.log(`Processing time: ${results.processing_time}s`);
    })
    .catch(error => console.error('Search error:', error));

// Batch search example
client.batchSearch([
    'machine learning',
    'neural networks',
    'deep learning'
]).then(results => {
    results.forEach((result, index) => {
        console.log(`Query ${index + 1}: ${result.total_results} results`);
    });
});
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "machine learning algorithms",
    "search_type": "hybrid",
    "max_results": 5,
    "include_synthesis": true
  }'

# Batch search
curl -X POST http://localhost:8000/search/batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "queries": ["AI research", "deep learning", "NLP"],
    "search_type": "vector"
  }'

# Upload and ingest file
curl -X POST http://localhost:8000/ingest/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"

# Create backup
curl -X POST http://localhost:8000/backup/create \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "backup_path": "/backups/cortex_backup",
    "include_images": true,
    "compress": true
  }'

# List backups
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/backup/list
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

### Error Response Format

```json
{
  "detail": "Error description",
  "status_code": 400,
  "timestamp": "2025-07-27T10:00:00Z",
  "path": "/search"
}
```

### Common Status Codes

- **200**: Success
- **400**: Bad Request - Invalid request data
- **401**: Unauthorized - Invalid or missing token
- **404**: Not Found - Resource not found
- **422**: Validation Error - Request validation failed
- **500**: Internal Server Error - Server-side error

### Error Handling Example

```python
try:
    result = client.search("machine learning")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 400:
        print("Bad request:", e.response.json()["detail"])
    elif e.response.status_code == 401:
        print("Authentication failed")
    elif e.response.status_code == 500:
        print("Server error:", e.response.json()["detail"])
    else:
        print(f"HTTP error {e.response.status_code}")
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Search endpoints**: 100 requests per minute
- **Ingestion endpoints**: 10 requests per minute
- **Backup endpoints**: 5 requests per minute

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1627392000
```

## Development

### Running in Development Mode

```bash
# Install development dependencies
pip install -r requirements.txt

# Start with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
python -m pytest tests/ -v

# Check API documentation
open http://localhost:8000/docs
```

### API Configuration

The API can be configured through environment variables or by modifying `api/main.py`:

```python
# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Custom authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Your authentication logic here
    pass
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Monitoring and Logging

The API includes built-in logging and monitoring:

```python
# Check logs
tail -f logs/api.log

# Monitor metrics (if configured)
curl http://localhost:8000/metrics
```

## Support

For issues, feature requests, or questions:

1. Check the [API documentation](http://localhost:8000/docs)
2. Review the [test suite](../tests/) for usage examples
3. Open an issue in the project repository

## License

This API is part of the Cortex Suite project. See the main project README for license information.