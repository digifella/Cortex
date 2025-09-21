#!/bin/bash
# Cortex Suite - One-Click Launcher
# This script handles everything automatically

set -e

echo ""
echo "==============================================="
echo "   CORTEX SUITE v4.8.0 (Search Stability & Docker Parity)"
echo "   Multi-Platform Support: Intel x86_64, Apple Silicon, ARM64"
echo "   GPU acceleration, timeout fixes, and improved reliability (2025-09-02)"
echo "   Date: $(date)"
echo "==============================================="
echo ""
echo "🚀 Starting Cortex Suite Launcher"

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

# Read any existing host mapping from .env
ENV_AI_DB_PATH=$(grep -E '^WINDOWS_AI_DATABASE_PATH=' .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'")
ENV_SOURCE_PATH=$(grep -E '^WINDOWS_KNOWLEDGE_SOURCE_PATH=' .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'")

# Offer interactive storage mapping (once) if not set in .env
ADD_AI_DB_MOUNT=""
ADD_SOURCE_MOUNT=""

if [ -z "$ENV_AI_DB_PATH" ]; then
    echo ""
    echo "📦 Storage Setup"
    echo "Would you like to store the AI database on a host folder (for easy backups)?"
    read -p "Use host folder for AI database? (y/N): " USE_HOST_AI_DB
    if [[ "$USE_HOST_AI_DB" =~ ^[Yy]$ ]]; then
        read -p "Enter host path for AI database (e.g., /Users/you/ai_databases): " HOST_AI_DB
        if [ -n "$HOST_AI_DB" ]; then
            # Persist to .env for future runs
            if grep -q '^WINDOWS_AI_DATABASE_PATH=' .env; then
                sed -i.bak "s|^WINDOWS_AI_DATABASE_PATH=.*|WINDOWS_AI_DATABASE_PATH=$HOST_AI_DB|" .env
            else
                echo "WINDOWS_AI_DATABASE_PATH=$HOST_AI_DB" >> .env
            fi
            ENV_AI_DB_PATH="$HOST_AI_DB"
        fi
    fi
fi

if [ -z "$ENV_SOURCE_PATH" ]; then
    read -p "Bind a host folder for Knowledge Source (documents to ingest)? (y/N): " USE_HOST_SRC
    if [[ "$USE_HOST_SRC" =~ ^[Yy]$ ]]; then
        read -p "Enter host path for Knowledge Source (e.g., /Users/you/Knowledge): " HOST_SRC
        if [ -n "$HOST_SRC" ]; then
            if grep -q '^WINDOWS_KNOWLEDGE_SOURCE_PATH=' .env; then
                sed -i.bak "s|^WINDOWS_KNOWLEDGE_SOURCE_PATH=.*|WINDOWS_KNOWLEDGE_SOURCE_PATH=$HOST_SRC|" .env
            else
                echo "WINDOWS_KNOWLEDGE_SOURCE_PATH=$HOST_SRC" >> .env
            fi
            ENV_SOURCE_PATH="$HOST_SRC"
        fi
    fi
fi

# Translate any configured host paths into -v mounts
if [ -n "$ENV_AI_DB_PATH" ]; then
    ADD_AI_DB_MOUNT="-v \"$ENV_AI_DB_PATH\":/data/ai_databases"
    echo "  📁 Mapping AI database to host: $ENV_AI_DB_PATH -> /data/ai_databases"
fi
if [ -n "$ENV_SOURCE_PATH" ]; then
    ADD_SOURCE_MOUNT="-v \"$ENV_SOURCE_PATH\":/data/knowledge_base:ro"
    echo "  📁 Mapping Knowledge Source to host (read-only): $ENV_SOURCE_PATH -> /data/knowledge_base"
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

# Detect user directories to mount
USER_VOLUME_MOUNTS=""
echo "🔍 Detecting user directories to mount..."

# Detect common user directories based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows (Git Bash/MSYS2)
    if [ -d "/c/Users" ]; then
        USER_VOLUME_MOUNTS="$USER_VOLUME_MOUNTS -v /c/Users:/mnt/c/Users:ro"
        echo "  📁 Mounting C:/Users as read-only"
    fi
    if [ -d "/d" ]; then
        USER_VOLUME_MOUNTS="$USER_VOLUME_MOUNTS -v /d:/mnt/d:ro"
        echo "  📁 Mounting D: drive as read-only"
    fi
    if [ -d "/e" ]; then
        USER_VOLUME_MOUNTS="$USER_VOLUME_MOUNTS -v /e:/mnt/e:ro"
        echo "  📁 Mounting E: drive as read-only"
    fi
    if [ -d "/f" ]; then
        USER_VOLUME_MOUNTS="$USER_VOLUME_MOUNTS -v /f:/mnt/f:ro"
        echo "  📁 Mounting F: drive as read-only"
    fi
elif [[ "$OSTYPE" == "linux-gnu" ]]; then
    # Linux/WSL
    if [ -d "/mnt/c" ]; then
        # WSL environment
        USER_VOLUME_MOUNTS="$USER_VOLUME_MOUNTS -v /mnt/c:/mnt/c:ro"
        echo "  📁 Mounting WSL /mnt/c as read-only"
        if [ -d "/mnt/d" ]; then
            USER_VOLUME_MOUNTS="$USER_VOLUME_MOUNTS -v /mnt/d:/mnt/d:ro"
            echo "  📁 Mounting WSL /mnt/d as read-only"
        fi
        if [ -d "/mnt/e" ]; then
            USER_VOLUME_MOUNTS="$USER_VOLUME_MOUNTS -v /mnt/e:/mnt/e:ro"
            echo "  📁 Mounting WSL /mnt/e as read-only"
        fi
        if [ -d "/mnt/f" ]; then
            USER_VOLUME_MOUNTS="$USER_VOLUME_MOUNTS -v /mnt/f:/mnt/f:ro"
            echo "  📁 Mounting WSL /mnt/f as read-only"
        fi
    else
        # Standard Linux
        if [ -d "$HOME" ]; then
            USER_VOLUME_MOUNTS="$USER_VOLUME_MOUNTS -v $HOME:/home/host_user:ro"
            echo "  📁 Mounting $HOME as /home/host_user (read-only)"
        fi
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if [ -d "$HOME" ]; then
        USER_VOLUME_MOUNTS="$USER_VOLUME_MOUNTS -v $HOME:/home/host_user:ro"
        echo "  📁 Mounting $HOME as /home/host_user (read-only)"
    fi
fi

# Add any user-defined paths from .env
if [ -f .env ]; then
    # Check for custom mount paths in .env file
    if grep -q "HOST_DOCUMENTS_PATH=" .env; then
        DOC_PATH=$(grep "HOST_DOCUMENTS_PATH=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
        if [ -n "$DOC_PATH" ] && [ -d "$DOC_PATH" ]; then
            USER_VOLUME_MOUNTS="$USER_VOLUME_MOUNTS -v \"$DOC_PATH\":/host_documents:ro"
            echo "  📁 Mounting custom documents path: $DOC_PATH"
        fi
    fi
fi

if [ -z "$USER_VOLUME_MOUNTS" ]; then
    echo "  ⚠️ No user directories detected for mounting"
    echo "  💡 You can manually add paths by editing the .env file"
else
    echo "  ✅ User directories will be available inside the container"
fi

# Check for NVIDIA GPU support before adding --gpus flag
echo "🔍 Checking for NVIDIA GPU support..."
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "🔍 Testing Docker GPU access..."
    if docker run --rm --gpus all hello-world >/dev/null 2>&1; then
        echo "✅ NVIDIA GPU detected and Docker GPU support available"
        GPU_FLAG="--gpus all"
    else
        echo "ℹ️ NVIDIA GPU detected but Docker GPU support unavailable, using CPU-only mode"
        GPU_FLAG=""
    fi
else
    echo "ℹ️ No NVIDIA GPU detected, using CPU-only mode"  
    echo "ℹ️ This is optimal for Apple Silicon M-series, ARM64 systems, and Intel CPUs without NVIDIA"
    GPU_FLAG=""
fi

# Run the container with intelligent GPU detection
echo "🚀 Starting Cortex Suite..."
docker run -d \
    --name cortex-suite \
    $GPU_FLAG \
    -p 8501:8501 \
    -p 8000:8000 \
    -v cortex_data:/data \
    $ADD_AI_DB_MOUNT \
    $ADD_SOURCE_MOUNT \
    -v cortex_logs:/home/cortex/app/logs \
    -v cortex_ollama:/home/cortex/.ollama \
    $USER_VOLUME_MOUNTS \
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
