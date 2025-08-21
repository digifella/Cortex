#!/bin/bash
# Cortex Suite - Enhanced Launcher with Model Management
# Includes automatic model checking and installation

set -e

echo "🚀 Cortex Suite - Enhanced Launcher with Model Management"
echo "========================================================"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"

# Function to check if model exists in Ollama
check_model() {
    local model_name=$1
    docker exec cortex-suite ollama list 2>/dev/null | grep -q "$model_name"
}

# Function to install model
install_model() {
    local model_name=$1
    local description=$2
    echo "📥 Installing $description ($model_name)..."
    echo "   This may take several minutes depending on model size..."
    
    if docker exec cortex-suite ollama pull "$model_name"; then
        echo "✅ $description installed successfully!"
    else
        echo "⚠️  Failed to install $description. You can install it later with:"
        echo "   docker exec cortex-suite ollama pull $model_name"
    fi
}

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^cortex-suite$"; then
    echo "📦 Cortex Suite container found"
    
    # Check if it's running
    if docker ps --format '{{.Names}}' | grep -q "^cortex-suite$"; then
        echo "✅ Cortex Suite is already running!"
        
        # Check models
        echo ""
        echo "🔍 Checking AI model status..."
        
        # Check required models
        if check_model "mistral"; then
            echo "✅ Base model (Mistral) is available"
        else
            echo "⚠️  Base model missing - installing..."
            install_model "mistral:7b-instruct-v0.3-q4_K_M" "Base Mistral model"
        fi
        
        # Check vision model
        if check_model "llava"; then
            echo "✅ Vision model (Llava) is available"
        else
            echo "📝 Vision model (Llava) not found"
            echo "   This is optional but recommended for image processing."
            read -p "   Install vision model? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                install_model "llava" "Vision model (Llava)"
            else
                echo "   You can install it later with: docker exec cortex-suite ollama pull llava"
            fi
        fi
        
        # Check enhanced proposal model
        if check_model "mistral-small"; then
            echo "✅ Enhanced proposal model (Mistral Small) is available"
        else
            echo "📝 Enhanced proposal model not found"
            echo "   This provides better proposal generation but is optional."
            read -p "   Install enhanced model? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                install_model "mistral-small3.2" "Enhanced proposal model"
            else
                echo "   You can install it later with: docker exec cortex-suite ollama pull mistral-small3.2"
            fi
        fi
        
        echo ""
        echo "🌐 Access your Cortex Suite at:"
        echo "   Main App: http://localhost:8501"
        echo "   API Docs: http://localhost:8000/docs"
        echo ""
        echo "💡 Check the System Status section in the sidebar for model availability"
        exit 0
    else
        echo "🔄 Starting existing Cortex Suite..."
        docker start cortex-suite
        echo "⏳ Waiting for services to start (30 seconds)..."
        sleep 30
    fi
else
    # First time setup
    echo "🆕 First time setup - this will take 5-10 minutes"
    echo "⏳ Please be patient while we:"
    echo "   • Build the Cortex Suite image"
    echo "   • Download AI models"
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
    if ! docker build -t cortex-suite -f Dockerfile ..; then
        echo "❌ Build failed! This could be due to:"
        echo "   • Network connectivity issues"
        echo "   • Insufficient disk space (need ~10GB)"
        echo "   • Docker permission issues"
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

    if [ -z "$USER_VOLUME_MOUNTS" ]; then
        echo "  ⚠️ No user directories detected for mounting"
    else
        echo "  ✅ User directories will be available inside the container"
    fi

    # Run the container
    echo "🚀 Starting Cortex Suite..."
    docker run -d \
        --name cortex-suite \
        -p 8501:8501 \
        -p 8000:8000 \
        -v cortex_data:/data \
        -v cortex_logs:/home/cortex/app/logs \
        $USER_VOLUME_MOUNTS \
        --env-file .env \
        --restart unless-stopped \
        cortex-suite

    if [ $? -ne 0 ]; then
        echo "❌ Failed to start Cortex Suite!"
        exit 1
    fi

    echo "⏳ Waiting for services to fully start (60 seconds)..."
    sleep 60
fi

# Now check and install models
echo ""
echo "🤖 Setting up AI models..."
echo "   Models will be downloaded as needed for full functionality."
echo ""

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama service to be ready..."
for i in {1..10}; do
    if docker exec cortex-suite ollama list >/dev/null 2>&1; then
        echo "✅ Ollama service is ready!"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "⚠️  Ollama service seems slow to start. Continuing anyway..."
        break
    fi
    sleep 3
done

# Install essential models
echo ""
echo "📥 Installing essential AI models..."

# Base model (required)
if ! check_model "mistral"; then
    install_model "mistral:7b-instruct-v0.3-q4_K_M" "Base Mistral model (required)"
fi

# Ask about optional models
echo ""
echo "📝 Optional model setup:"
echo "   The following models enhance functionality but are not required."
echo ""

# Vision model for image processing
if ! check_model "llava"; then
    echo "🖼️  Vision Model (Llava) - Enables image description and analysis"
    echo "   Size: ~4GB | Use case: Document images, charts, diagrams"
    read -p "   Install vision model? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        install_model "llava" "Vision model (Llava)"
    fi
fi

# Enhanced proposal model
if ! check_model "mistral-small"; then
    echo ""
    echo "📝 Enhanced Proposal Model (Mistral Small 3.2) - Better proposal generation"
    echo "   Size: ~7GB | Use case: Higher quality proposals and analysis"
    read -p "   Install enhanced model? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        install_model "mistral-small3.2" "Enhanced proposal model"
    fi
fi

# Final checks
echo ""
echo "🔍 Final system check..."

# Check if services are responding
if curl -s http://localhost:8501/_stcore/health >/dev/null 2>&1; then
    echo "✅ Streamlit UI is ready!"
else
    echo "⚠️  UI might still be starting up (give it another minute)"
fi

if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ API is ready!"
else
    echo "⚠️  API might still be starting up"
fi

echo ""
echo "🎉 Cortex Suite setup complete!"
echo ""
echo "🌐 Access your Cortex Suite at:"
echo "   Main Application: http://localhost:8501"
echo "   API Documentation: http://localhost:8000/docs"
echo ""
echo "💡 Tips:"
echo "   • Check the 'System Status' in the main page sidebar"
echo "   • Missing models? Install them with: docker exec cortex-suite ollama pull MODEL_NAME"
echo "   • Need help? Look for the Help system in the sidebar"
echo ""
echo "📋 Management commands:"
echo "   Stop:    docker stop cortex-suite"
echo "   Start:   docker start cortex-suite"
echo "   Logs:    docker logs cortex-suite -f"
echo "   Remove:  docker stop cortex-suite && docker rm cortex-suite"
echo ""
echo "🔧 Model management:"
echo "   List:    docker exec cortex-suite ollama list"
echo "   Install: docker exec cortex-suite ollama pull MODEL_NAME"
echo "   Remove:  docker exec cortex-suite ollama rm MODEL_NAME"