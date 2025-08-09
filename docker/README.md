# Cortex Suite Docker Deployment Guide

This directory contains multiple Docker deployment options for the Cortex Suite, from simple single-container setups to production-ready multi-service architectures.

## 🚀 Quick Start

### Option 1: Single Container (Simplest)
Perfect for development, testing, or single-user setups.

```bash
cd docker
cp .env.example .env
# Edit .env with your configuration
./deployment-scripts/deploy.sh single
```

**Access:**
- Streamlit UI: http://localhost:8501
- API: http://localhost:8000

### Option 2: Multi-Container (Recommended)
Scalable setup with separate services for better resource management.

```bash
cd docker
cp .env.example .env
# Edit .env with your configuration
./deployment-scripts/deploy.sh development
```

**Access:**
- Streamlit UI: http://localhost:8501
- API: http://localhost:8000
- ChromaDB: http://localhost:8001

### Option 3: Production (Enterprise)
Full production setup with load balancing, monitoring, and high availability.

```bash
cd docker
cp .env.example .env.prod
# Edit .env.prod with production configuration
./deployment-scripts/deploy.sh production
```

**Access:**
- Application: https://your-domain.com
- Grafana Monitoring: http://localhost:3000
- Prometheus Metrics: http://localhost:9090

## 📁 File Structure

```
docker/
├── Dockerfile                     # Single container (all-in-one)
├── docker-compose.yml            # Multi-container development
├── docker-compose.prod.yml       # Production setup
├── Dockerfile.api                # API service container
├── Dockerfile.ui                 # UI service container
├── .env.example                  # Environment template
├── .dockerignore                 # Docker ignore rules
├── deployment-scripts/
│   └── deploy.sh                 # Automated deployment script
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

**Required:**
```bash
LLM_PROVIDER=ollama              # or "openai", "gemini"
OLLAMA_MODEL=mistral:7b-instruct-v0.3-q4_K_M
AI_DATABASE_PATH=/data/ai_databases
KNOWLEDGE_SOURCE_PATH=/data/knowledge_base
```

**Optional API Keys:**
```bash
OPENAI_API_KEY=your_key          # If using OpenAI
GEMINI_API_KEY=your_key          # If using Gemini
YOUTUBE_API_KEY=your_key         # For research features
```

### GPU Support

For NVIDIA GPU acceleration with Ollama:

```bash
# In .env file
ENABLE_GPU=true

# Deploy with GPU profile
docker-compose --profile gpu up -d
```

## 🛠️ Deployment Options Explained

### Single Container (`Dockerfile`)

**Pros:**
- Simplest setup
- All services in one container
- Perfect for development/testing
- Minimal resource overhead

**Cons:**
- No service isolation
- Harder to scale individual components
- Single point of failure

**Use Cases:**
- Personal use
- Development environment
- Small teams
- Resource-constrained environments

### Multi-Container (`docker-compose.yml`)

**Architecture:**
- **Ollama Service**: Local LLM processing
- **ChromaDB**: Vector database
- **Cortex API**: Backend REST API
- **Cortex UI**: Streamlit frontend
- **Model Init**: One-time model download

**Pros:**
- Service isolation
- Independent scaling
- Better resource management
- Easy to maintain

**Cons:**
- Slightly more complex
- More containers to manage

**Use Cases:**
- Development teams
- Small to medium deployments
- Learning Docker concepts

### Production Setup (`docker-compose.prod.yml`)

**Architecture:**
- **Nginx**: Reverse proxy & load balancer
- **PostgreSQL**: Persistent metadata storage
- **Redis**: Caching & session management
- **Multiple API instances**: Load balanced
- **Multiple UI instances**: High availability
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards
- **Backup Service**: Automated backups

**Pros:**
- High availability
- Load balancing
- Comprehensive monitoring
- Automated backups
- Production-ready security

**Cons:**
- Complex setup
- Higher resource requirements
- Requires operational knowledge

**Use Cases:**
- Production deployments
- Enterprise environments
- High-traffic scenarios
- Mission-critical applications

## 📊 Resource Requirements

### Single Container
- **CPU**: 2 cores minimum
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB for base system + data
- **GPU**: Optional (NVIDIA for local LLM acceleration)

### Multi-Container Development
- **CPU**: 4 cores minimum
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB for services + data
- **GPU**: Optional

### Production Setup
- **CPU**: 8 cores minimum
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB+ for services + data + backups
- **GPU**: Recommended for LLM processing
- **Network**: High bandwidth for API traffic

## 🔄 Management Commands

### Starting Services
```bash
# Development
./deployment-scripts/deploy.sh development

# Production
./deployment-scripts/deploy.sh production

# Single container
./deployment-scripts/deploy.sh single
```

### Monitoring
```bash
# View logs
docker-compose logs -f

# Check service status
docker-compose ps

# Resource usage
docker stats
```

### Maintenance
```bash
# Update services
docker-compose pull
docker-compose up -d

# Restart specific service
docker-compose restart cortex-api

# Scale service (development)
docker-compose up -d --scale cortex-api=3
```

### Backup & Recovery
```bash
# Manual backup (built into containers)
docker exec cortex-api python -c "
from cortex_engine.sync_backup_manager import SyncBackupManager
manager = SyncBackupManager('/data/ai_databases')
backup = manager.create_backup('manual_backup')
print(f'Backup created: {backup.backup_id}')
"

# Volume backup
docker run --rm -v cortex_data:/data -v $(pwd):/backup alpine tar czf /backup/cortex_backup.tar.gz /data
```

## 🚨 Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check logs
docker-compose logs service_name

# Check disk space
df -h

# Check ports
netstat -tulpn | grep :8501
```

**Ollama models not downloading:**
```bash
# Manual model pull
docker exec cortex-ollama ollama pull mistral:7b-instruct-v0.3-q4_K_M

# Check Ollama logs
docker-compose logs ollama
```

**ChromaDB connection issues:**
```bash
# Test ChromaDB connectivity
curl http://localhost:8001/api/v1/heartbeat

# Reset ChromaDB data
docker-compose down
docker volume rm cortex_chroma_data
docker-compose up -d
```

**Memory issues:**
```bash
# Check container memory usage
docker stats

# Increase Docker memory limits
# Edit docker-compose.yml deploy.resources sections
```

### Performance Tuning

**For CPU-intensive workloads:**
```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '4.0'
    reservations:
      cpus: '2.0'
```

**For memory optimization:**
```yaml
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G
```

## 🔐 Security Considerations

### Development
- Uses HTTP only
- Default credentials
- All ports exposed
- Minimal authentication

### Production
- HTTPS with SSL certificates
- Strong passwords in `.env.prod`
- Limited port exposure
- JWT authentication
- Network isolation
- Regular security updates

### Securing Production Deployment

1. **SSL Certificates:**
   ```bash
   # Add your SSL certificates to docker/ssl/
   cp your_cert.pem docker/ssl/
   cp your_key.pem docker/ssl/
   ```

2. **Strong Passwords:**
   ```bash
   # Generate secure passwords
   openssl rand -base64 32
   ```

3. **Firewall Rules:**
   ```bash
   # Only allow necessary ports
   ufw allow 80/tcp
   ufw allow 443/tcp
   ufw deny 8000/tcp  # Block direct API access
   ```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale API instances
docker-compose up -d --scale cortex-api=3

# Scale UI instances  
docker-compose up -d --scale cortex-ui=2
```

### Load Balancing
Production setup includes Nginx load balancer with:
- Round-robin distribution
- Health checks
- Automatic failover
- Session persistence

### Database Scaling
For high-volume deployments:
- Use external ChromaDB cluster
- Implement PostgreSQL read replicas
- Add Redis clustering
- Consider database sharding

## 🔄 Updates & Maintenance

### Regular Updates
```bash
# Update base images
docker-compose pull

# Rebuild custom images
docker-compose build

# Rolling update
docker-compose up -d --no-deps --build service_name
```

### Backup Strategy
- **Automated**: Built-in backup service (production)
- **Manual**: Volume snapshots
- **Database**: PostgreSQL dumps
- **Configuration**: Git repository backups

This comprehensive Docker setup provides flexibility from simple development environments to enterprise-grade production deployments, ensuring the Cortex Suite can scale with your needs.