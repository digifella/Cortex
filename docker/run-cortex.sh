#!/bin/bash
# Cortex Suite - One-Click Launcher
# This script handles everything automatically

set -e

echo "🚀 Cortex Suite - Easy Launcher"
echo "================================"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^cortex-suite$"; then
    echo "📦 Cortex Suite container found"
    
    # Check if it's running
    if docker ps --format '{{.Names}}' | grep -q "^cortex-suite$"; then
        echo "✅ Cortex Suite is already running!"
        echo ""
        echo "🌐 Access your Cortex Suite at:"
        echo "   Main App: http://localhost:8501"
        echo "   API Docs: http://localhost:8000/docs"
        exit 0
    else
        echo "🔄 Starting existing Cortex Suite..."
        docker start cortex-suite
        echo "⏳ Waiting for services to start (30 seconds)..."
        sleep 30
        echo "✅ Cortex Suite is now running!"
        echo ""
        echo "🌐 Access your Cortex Suite at:"
        echo "   Main App: http://localhost:8501"
        echo "   API Docs: http://localhost:8000/docs"
        exit 0
    fi
fi

# First time setup
echo "🆕 First time setup - this will take 5-10 minutes"
echo "⏳ Please be patient while we:"
echo "   • Build the Cortex Suite image"
echo "   • Download AI models (~4GB)"
echo "   • Set up the database"
echo ""

# Check if .env exists, create from example if not
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "📋 Created .env configuration file"
    else
        echo "❌ .env.example not found! Make sure you're in the docker directory."
        exit 1
    fi
fi

# Build the image
echo "🔨 Building Cortex Suite (this may take a while)..."
echo "    This includes downloading Python packages and system dependencies..."
if ! docker build -t cortex-suite -f Dockerfile .; then
    echo "❌ Build failed! This could be due to:"
    echo "   • Network connectivity issues"
    echo "   • Insufficient disk space (need ~10GB)"
    echo "   • Docker permission issues"
    echo ""
    echo "Try running: docker system prune -f"
    echo "Then try again."
    exit 1
fi

# Run the container
echo "🚀 Starting Cortex Suite..."
docker run -d \
    --name cortex-suite \
    -p 8501:8501 \
    -p 8000:8000 \
    -v cortex_data:/data \
    -v cortex_logs:/home/cortex/app/logs \
    -v cortex_ollama:/home/cortex/.ollama \
    --env-file .env \
    --restart unless-stopped \
    cortex-suite

if [ $? -ne 0 ]; then
    echo "❌ Failed to start Cortex Suite!"
    echo "This might be because the ports are already in use."
    echo "Try running: docker stop cortex-suite && docker rm cortex-suite"
    exit 1
fi

echo "⏳ Waiting for services to fully start (60 seconds)..."
echo "   This includes downloading and setting up AI models..."

# Wait and show progress
for i in {1..12}; do
    sleep 5
    echo "   ... $((i*5)) seconds elapsed"
done

# Check if services are responding
echo "🔍 Checking if services are ready..."

# Simple health check
if curl -s http://localhost:8501/_stcore/health >/dev/null 2>&1; then
    echo "✅ Streamlit UI is ready!"
else
    echo "⚠️  UI might still be starting up..."
fi

if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ API is ready!"
else
    echo "⚠️  API might still be starting up..."
fi

echo ""
echo "🎉 Cortex Suite is now running!"
echo ""
echo "🌐 Access your Cortex Suite at:"
echo "   Main Application: http://localhost:8501"
echo "   API Documentation: http://localhost:8000/docs"
echo ""
echo "📋 Useful commands:"
echo "   Stop:    docker stop cortex-suite"
echo "   Start:   docker start cortex-suite"
echo "   Logs:    docker logs cortex-suite -f"
echo "   Remove:  docker stop cortex-suite && docker rm cortex-suite"
echo ""
echo "💡 Your data is safely stored in Docker volumes and will persist"
echo "   between stops and starts!"
echo ""
echo "📦 NOTE: AI models are downloading in the background. The interface is"
echo "   accessible immediately, with full AI features activating as"
echo "   downloads complete (15-30 minutes for first-time setup)."