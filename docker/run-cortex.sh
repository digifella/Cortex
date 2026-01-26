#!/bin/bash
# Cortex Suite - One-Click Launcher
# This script handles everything automatically

set -e

normalize_path_for_docker() {
    local input="$1"
    if [ -z "$input" ]; then
        echo ""
        return
    fi

    # Prefer wslpath when available (WSL/git-bash with WSL)
    if command -v wslpath >/dev/null 2>&1; then
        # wslpath handles both Windows and WSL-style inputs
        local converted
        converted=$(wslpath -u "$input" 2>/dev/null || true)
        if [ -n "$converted" ]; then
            echo "$converted"
            return
        fi
    fi

    # Fallback: convert Windows drive paths (e.g., C:\path or C:/path) to /c/path
    if [[ "$input" =~ ^[A-Za-z]:[\\/] ]]; then
        local drive rest
        drive="${input:0:1}"
        rest="${input:2}"
        rest="${rest//\\//}"
        echo "/${drive,,}/${rest}"
        return
    fi

    # Otherwise return as-is (already POSIX/WSL)
    echo "$input"
}

echo ""
echo "==============================================="
echo "   CORTEX SUITE v5.6.0 - Docker Installer"
echo "   Multi-Platform Support: Intel x86_64, Apple Silicon, ARM64"
echo "   GPU acceleration and improved reliability"
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
        echo "🔧 Fixing data directory permissions..."
        docker exec -u root cortex-suite chown -R cortex:cortex /data 2>/dev/null || echo "   Permission fix completed"
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
ENV_AI_DB_PATH_RAW=$(grep -E '^WINDOWS_AI_DATABASE_PATH=' .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'")
ENV_SOURCE_PATH_RAW=$(grep -E '^WINDOWS_KNOWLEDGE_SOURCE_PATH=' .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'")
ENV_AI_DB_PATH=$(normalize_path_for_docker "$ENV_AI_DB_PATH_RAW")
ENV_SOURCE_PATH=$(normalize_path_for_docker "$ENV_SOURCE_PATH_RAW")

# Show existing configuration if present
if [ -n "$ENV_AI_DB_PATH" ] || [ -n "$ENV_SOURCE_PATH" ]; then
    echo ""
    echo "ℹ️  Found existing storage configuration:"
    [ -n "$ENV_AI_DB_PATH" ] && echo "   AI Database: $ENV_AI_DB_PATH"
    [ -n "$ENV_SOURCE_PATH" ] && echo "   Knowledge Source: $ENV_SOURCE_PATH"
    echo ""
fi

# ALWAYS prompt for AI Database Path (with default if exists)
enter_ai_db_path() {
    echo ""
    echo "==================================================================="
    echo "STORAGE CONFIGURATION: AI Database Path"
    echo "==================================================================="
    echo "This is where Cortex will store the knowledge graph and vector DB."
    echo "IMPORTANT: Configure paths NOW before the long build process."
    echo ""
    if [ -n "$ENV_AI_DB_PATH" ]; then
        echo "Current: $ENV_AI_DB_PATH"
        echo ""
    fi
    read -p "Use host folder for AI database storage? (y/N): " USE_HOST_AI_DB
    if [[ ! "$USE_HOST_AI_DB" =~ ^[Yy]$ ]]; then
        return
    fi

    while true; do
        if [ -n "$ENV_AI_DB_PATH" ]; then
            read -p "Enter host path [$ENV_AI_DB_PATH]: " HOST_AI_DB
        else
            read -p "Enter host path (e.g., /Users/you/ai_databases or ~/ai_databases): " HOST_AI_DB
        fi

        # Use existing path if user just pressed ENTER
        if [ -z "$HOST_AI_DB" ] && [ -n "$ENV_AI_DB_PATH" ]; then
            HOST_AI_DB="$ENV_AI_DB_PATH"
            echo "ℹ️  Using existing path: $HOST_AI_DB"
        fi

        if [ -z "$HOST_AI_DB" ]; then
            echo "ℹ️  Skipping AI database host mapping..."
            return
        fi

        # Expand ~ to home directory
        HOST_AI_DB="${HOST_AI_DB/#\~/$HOME}"
        HOST_AI_DB=$(normalize_path_for_docker "$HOST_AI_DB")

        # Validate the path - check if parent directory exists or can be created
        PARENT_DIR=$(dirname "$HOST_AI_DB")
        if [ ! -d "$PARENT_DIR" ]; then
            echo ""
            echo "*** ERROR: Parent directory does not exist! ***"
            echo "Path: $PARENT_DIR"
            echo ""
            read -p "Would you like to enter a different path? (Y/n): " RETRY_AI_DB
            RETRY_AI_DB=${RETRY_AI_DB:-Y}
            if [[ "$RETRY_AI_DB" =~ ^[Nn]$ ]]; then
                echo "ℹ️  Skipping AI database host mapping..."
                return
            fi
            continue
        fi

        # Try to create directory if it doesn't exist
        if [ ! -d "$HOST_AI_DB" ]; then
            echo "ℹ️  Directory does not exist, attempting to create..."
            if ! mkdir -p "$HOST_AI_DB" 2>/dev/null; then
                echo ""
                echo "*** ERROR: Failed to create directory ***"
                echo "Path: $HOST_AI_DB"
                echo ""
                echo "Possible causes:"
                echo "  - Permission denied"
                echo "  - Invalid path format"
                echo ""
                read -p "Would you like to enter a different path? (Y/n): " RETRY_AI_DB
                RETRY_AI_DB=${RETRY_AI_DB:-Y}
                if [[ "$RETRY_AI_DB" =~ ^[Nn]$ ]]; then
                    echo "ℹ️  Skipping AI database host mapping..."
                    return
                fi
                continue
            fi
            echo "✅ Directory created successfully!"
        fi

        # Path is valid - save to .env
        echo "✅ AI database path validated: $HOST_AI_DB"
        # Remove old entries and add new ones
        grep -v -E '^(WINDOWS_AI_DATABASE_PATH=|AI_DATABASE_PATH=)' .env > .env.tmp 2>/dev/null || true
        mv .env.tmp .env 2>/dev/null || true
        echo "WINDOWS_AI_DATABASE_PATH=$HOST_AI_DB" >> .env
        echo "AI_DATABASE_PATH=/data/ai_databases" >> .env
        ENV_AI_DB_PATH="$HOST_AI_DB"
        break
    done
}
enter_ai_db_path

# ALWAYS prompt for Knowledge Source Path (with default if exists)
enter_source_path() {
    echo ""
    echo "==================================================================="
    echo "STORAGE CONFIGURATION: Knowledge Source Path"
    echo "==================================================================="
    echo "This is where your source documents are stored for ingestion."
    echo "(PDF, Word, text files, etc.)"
    echo ""
    if [ -n "$ENV_SOURCE_PATH" ]; then
        echo "Current: $ENV_SOURCE_PATH"
        echo ""
    fi
    read -p "Use host folder for Knowledge Source documents? (y/N): " USE_HOST_SRC
    if [[ ! "$USE_HOST_SRC" =~ ^[Yy]$ ]]; then
        return
    fi

    while true; do
        if [ -n "$ENV_SOURCE_PATH" ]; then
            read -p "Enter host path [$ENV_SOURCE_PATH]: " HOST_SRC
        else
            read -p "Enter host path (e.g., /Users/you/Documents/KB_Source or ~/Documents): " HOST_SRC
        fi

        # Use existing path if user just pressed ENTER
        if [ -z "$HOST_SRC" ] && [ -n "$ENV_SOURCE_PATH" ]; then
            HOST_SRC="$ENV_SOURCE_PATH"
            echo "ℹ️  Using existing path: $HOST_SRC"
        fi

        if [ -z "$HOST_SRC" ]; then
            echo "ℹ️  Skipping Knowledge Source host mapping..."
            return
        fi

        # Expand ~ to home directory
        HOST_SRC="${HOST_SRC/#\~/$HOME}"
        HOST_SRC=$(normalize_path_for_docker "$HOST_SRC")

        # Validate the path - check if parent directory exists
        PARENT_DIR=$(dirname "$HOST_SRC")
        if [ ! -d "$PARENT_DIR" ]; then
            echo ""
            echo "*** ERROR: Parent directory does not exist! ***"
            echo "Path: $PARENT_DIR"
            echo ""
            read -p "Would you like to enter a different path? (Y/n): " RETRY_SRC
            RETRY_SRC=${RETRY_SRC:-Y}
            if [[ "$RETRY_SRC" =~ ^[Nn]$ ]]; then
                echo "ℹ️  Skipping Knowledge Source host mapping..."
                return
            fi
            continue
        fi

        # Verify or create source directory
        if [ ! -d "$HOST_SRC" ]; then
            echo "⚠️  WARNING: Source directory does not exist"
            echo "Path: $HOST_SRC"
            echo ""
            read -p "Create this directory? (y/N): " CREATE_SRC
            if [[ ! "$CREATE_SRC" =~ ^[Yy]$ ]]; then
                read -p "Would you like to enter a different path? (Y/n): " RETRY_SRC
                RETRY_SRC=${RETRY_SRC:-Y}
                if [[ "$RETRY_SRC" =~ ^[Nn]$ ]]; then
                    echo "ℹ️  Skipping Knowledge Source host mapping..."
                    return
                fi
                continue
            fi
            if ! mkdir -p "$HOST_SRC" 2>/dev/null; then
                echo ""
                echo "*** ERROR: Failed to create directory ***"
                echo ""
                read -p "Would you like to enter a different path? (Y/n): " RETRY_SRC
                RETRY_SRC=${RETRY_SRC:-Y}
                if [[ "$RETRY_SRC" =~ ^[Nn]$ ]]; then
                    echo "ℹ️  Skipping Knowledge Source host mapping..."
                    return
                fi
                continue
            fi
            echo "✅ Directory created successfully!"
        fi

        # Path is valid - save to .env
        echo "✅ Knowledge Source path validated: $HOST_SRC"
        # Remove old entries and add new ones
        grep -v -E '^(WINDOWS_KNOWLEDGE_SOURCE_PATH=|KNOWLEDGE_SOURCE_PATH=)' .env > .env.tmp 2>/dev/null || true
        mv .env.tmp .env 2>/dev/null || true
        echo "WINDOWS_KNOWLEDGE_SOURCE_PATH=$HOST_SRC" >> .env
        echo "KNOWLEDGE_SOURCE_PATH=/data/knowledge_base" >> .env
        ENV_SOURCE_PATH="$HOST_SRC"
        break
    done
}
enter_source_path

# Translate any configured host paths into -v mounts
# Note: Paths are stored in arrays to handle spaces correctly
AI_DB_MOUNT_ARGS=()
SOURCE_MOUNT_ARGS=()

if [ -n "$ENV_AI_DB_PATH" ]; then
    # Create directory if it doesn't exist
    mkdir -p "$ENV_AI_DB_PATH" 2>/dev/null || true
    AI_DB_MOUNT_ARGS=(-v "$ENV_AI_DB_PATH:/data/ai_databases")
fi
if [ -n "$ENV_SOURCE_PATH" ]; then
    # Create directory if it doesn't exist
    mkdir -p "$ENV_SOURCE_PATH" 2>/dev/null || true
    SOURCE_MOUNT_ARGS=(-v "$ENV_SOURCE_PATH:/data/knowledge_base:ro")
fi

# Show final configuration summary
echo ""
echo "==================================================================="
echo "CONFIGURATION SUMMARY"
echo "==================================================================="
if [ -n "$ENV_AI_DB_PATH" ]; then
    echo "  📁 AI Database:      $ENV_AI_DB_PATH -> /data/ai_databases"
else
    echo "  📁 AI Database:      [Using Docker volume - data inside container]"
fi
if [ -n "$ENV_SOURCE_PATH" ]; then
    echo "  📁 Knowledge Source: $ENV_SOURCE_PATH -> /data/knowledge_base (read-only)"
else
    echo "  📁 Knowledge Source: [Using Docker volume - data inside container]"
fi
echo ""

# Final confirmation before starting the long build
echo "==================================================================="
echo "READY TO BUILD"
echo "==================================================================="
echo "The Docker build process will now begin."
echo "This typically takes 5-10 minutes depending on your internet speed."
echo ""
read -p "Proceed with build? (Y/n): " CONFIRM_BUILD
CONFIRM_BUILD=${CONFIRM_BUILD:-Y}
if [[ "$CONFIRM_BUILD" =~ ^[Nn]$ ]]; then
    echo ""
    echo "Build cancelled. Run this script again when ready."
    exit 0
fi

# Build the image
echo ""
echo "🔨 Building Cortex Suite (this may take a while)..."
echo "    This includes downloading Python packages and system dependencies..."

# Detect GPU and build appropriate image
echo "🔍 Checking for NVIDIA GPU..."
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "✅ NVIDIA GPU detected - Building GPU-accelerated image"
    echo "🔨 Using Dockerfile.gpu for CUDA support"
    DOCKERFILE="Dockerfile.gpu"
else
    echo "ℹ️ No NVIDIA GPU detected - Building CPU-only image"
    echo "🔨 Using standard Dockerfile"
    DOCKERFILE="Dockerfile"
fi

if ! docker build -t cortex-suite -f $DOCKERFILE .; then
    echo "❌ Build failed! This could be due to:"
    echo "   • Network connectivity issues (download interrupted)"
    echo "   • Insufficient disk space (need ~10GB)"
    echo "   • Docker permission issues"
    echo "   • Corrupted Docker build cache"
    echo ""
    echo "🔧 RECOMMENDED FIX - Run these commands then retry:"
    echo "   docker system prune -a -f"
    echo "   docker builder prune -a -f"
    echo ""
    echo "💡 If error mentions 'short read' or 'unexpected EOF':"
    echo "   This is a network/download issue - just retry the build"
    echo "   The download was interrupted and needs to restart"
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
            USER_VOLUME_MOUNTS="$USER_VOLUME_MOUNTS -v $DOC_PATH:/host_documents:ro"
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

# Build docker run command with proper quoting for paths with spaces
DOCKER_RUN_CMD=(docker run -d --name cortex-suite)

# Add GPU flag if available
if [ -n "$GPU_FLAG" ]; then
    DOCKER_RUN_CMD+=($GPU_FLAG)
fi

# Add port mappings
DOCKER_RUN_CMD+=(-p 8501:8501 -p 8000:8000)

# Add base volumes
DOCKER_RUN_CMD+=(-v cortex_data:/data)

# Add custom mounts (arrays preserve spaces in paths)
if [ ${#AI_DB_MOUNT_ARGS[@]} -gt 0 ]; then
    DOCKER_RUN_CMD+=("${AI_DB_MOUNT_ARGS[@]}")
fi
if [ ${#SOURCE_MOUNT_ARGS[@]} -gt 0 ]; then
    DOCKER_RUN_CMD+=("${SOURCE_MOUNT_ARGS[@]}")
fi

# Add remaining volumes
DOCKER_RUN_CMD+=(-v cortex_logs:/home/cortex/app/logs -v cortex_ollama:/home/cortex/.ollama)

# Add user volume mounts
if [ -n "$USER_VOLUME_MOUNTS" ]; then
    DOCKER_RUN_CMD+=($USER_VOLUME_MOUNTS)
fi

# Add env file and restart policy
DOCKER_RUN_CMD+=(--env-file .env --restart unless-stopped cortex-suite)

# Execute the command
"${DOCKER_RUN_CMD[@]}"

if [ $? -ne 0 ]; then
    echo "❌ Failed to start Cortex Suite!"
    echo "This might be because the ports are already in use."
    echo "Try running: docker stop cortex-suite && docker rm cortex-suite"
    exit 1
fi

echo "🔧 Fixing data directory permissions..."
docker exec -u root cortex-suite chown -R cortex:cortex /data 2>/dev/null || echo "   Permission fix completed"

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
