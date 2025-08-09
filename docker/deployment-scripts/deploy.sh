#!/bin/bash
# Cortex Suite Deployment Script

set -e

# Configuration
DEPLOY_TYPE=${1:-"development"}
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

case $DEPLOY_TYPE in
    "development"|"dev")
        COMPOSE_FILE="docker-compose.yml"
        ENV_FILE=".env"
        echo "🚀 Deploying in DEVELOPMENT mode"
        ;;
    "production"|"prod")
        COMPOSE_FILE="docker-compose.prod.yml"
        ENV_FILE=".env.prod"
        echo "🚀 Deploying in PRODUCTION mode"
        ;;
    "single"|"simple")
        echo "🚀 Deploying SINGLE CONTAINER mode"
        docker build -t cortex-suite -f Dockerfile ..
        docker run -d \
            --name cortex-suite \
            -p 8501:8501 \
            -p 8000:8000 \
            -v cortex_data:/home/cortex/data \
            -v cortex_logs:/home/cortex/app/logs \
            --env-file .env \
            cortex-suite
        echo "✅ Single container deployment complete"
        echo "🌐 Streamlit UI: http://localhost:8501"
        echo "🔗 API: http://localhost:8000"
        exit 0
        ;;
    *)
        echo "❌ Invalid deployment type: $DEPLOY_TYPE"
        echo "Usage: $0 [development|production|single]"
        exit 1
        ;;
esac

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Environment file $ENV_FILE not found"
    echo "Please create $ENV_FILE with required environment variables"
    exit 1
fi

# Check if Docker Compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ Docker Compose file $COMPOSE_FILE not found"
    exit 1
fi

echo "📋 Using configuration:"
echo "   Compose file: $COMPOSE_FILE"
echo "   Environment: $ENV_FILE"

# Create required directories
echo "📁 Creating required directories..."
mkdir -p data/ai_databases data/knowledge_base logs backups

# Pull latest images
echo "⬇️ Pulling latest Docker images..."
docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull

# Build custom images
echo "🔨 Building Cortex Suite images..."
docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build

# Start services
echo "🚀 Starting Cortex Suite services..."
if [ "$DEPLOY_TYPE" = "production" ] || [ "$DEPLOY_TYPE" = "prod" ]; then
    # Production deployment with specific profiles
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --remove-orphans
else
    # Development deployment
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --remove-orphans
fi

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
if [ "$DEPLOY_TYPE" = "production" ] || [ "$DEPLOY_TYPE" = "prod" ]; then
    services=("nginx" "postgres" "redis" "chromadb" "cortex-api-1" "cortex-ui-1")
else
    services=("ollama" "chromadb" "cortex-api" "cortex-ui")
fi

for service in "${services[@]}"; do
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps "$service" | grep -q "Up (healthy)"; then
        echo "✅ $service is healthy"
    else
        echo "⚠️ $service may not be ready yet"
    fi
done

# Initialize models (for development)
if [ "$DEPLOY_TYPE" = "development" ] || [ "$DEPLOY_TYPE" = "dev" ]; then
    echo "🤖 Initializing AI models..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up model-init
fi

echo ""
echo "🎉 Cortex Suite deployment complete!"
echo ""
if [ "$DEPLOY_TYPE" = "production" ] || [ "$DEPLOY_TYPE" = "prod" ]; then
    echo "🌐 Application: https://your-domain.com"
    echo "📊 Monitoring: http://localhost:3000 (Grafana)"
    echo "📈 Metrics: http://localhost:9090 (Prometheus)"
else
    echo "🌐 Streamlit UI: http://localhost:8501"
    echo "🔗 API Documentation: http://localhost:8000/docs"
    echo "📊 API Health: http://localhost:8000/health"
fi
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker-compose -f $COMPOSE_FILE logs -f"
echo "   Stop services: docker-compose -f $COMPOSE_FILE down"
echo "   Update: ./deploy.sh $DEPLOY_TYPE"