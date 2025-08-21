#!/bin/bash
# Cortex Suite - Hybrid Model Architecture Launcher
# Supports both Docker Model Runner and Ollama backends

set -e

echo "🚀 Cortex Suite - Hybrid Model Architecture Launcher"
echo "=================================================="

# Configuration
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-docker-compose-hybrid.yml}"
MODEL_STRATEGY="${MODEL_STRATEGY:-hybrid_docker_preferred}"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed!"
        echo "Please install Docker Desktop and try again."
        echo "https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running!"
        echo "Please start Docker Desktop and try again."
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version >/dev/null 2>&1; then
        log_error "Docker Compose is not available!"
        echo "Please ensure you have Docker Compose v2 installed."
        exit 1
    fi
    
    # Check Docker Model Runner support (optional)
    if docker model --help >/dev/null 2>&1; then
        log_success "Docker Model Runner detected - enterprise features available"
        export DOCKER_MODELS_AVAILABLE=true
    else
        log_warning "Docker Model Runner not available - using Ollama fallback"
        export DOCKER_MODELS_AVAILABLE=false
    fi
    
    log_success "Prerequisites check complete"
}

# Function to check available disk space
check_disk_space() {
    log_info "Checking available disk space..."
    
    # Get available space in GB
    available_space=$(df . | awk 'NR==2 {print int($4/1024/1024)}')
    
    if [ "$available_space" -lt 15 ]; then
        log_warning "Only ${available_space}GB available. Recommend at least 15GB for full installation."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_success "${available_space}GB available - sufficient for installation"
    fi
}

# Function to create environment file
setup_environment() {
    log_info "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            log_success "Created .env from template"
        else
            log_error ".env.example not found!"
            exit 1
        fi
    else
        log_info ".env file already exists"
    fi
    
    # Update environment variables
    sed -i "s/MODEL_DISTRIBUTION_STRATEGY=.*/MODEL_DISTRIBUTION_STRATEGY=${MODEL_STRATEGY}/" .env
    sed -i "s/DEPLOYMENT_ENV=.*/DEPLOYMENT_ENV=${DEPLOYMENT_ENV}/" .env
    
    # Enable Docker models if available
    if [ "$DOCKER_MODELS_AVAILABLE" = "true" ]; then
        sed -i "s/ENABLE_DOCKER_MODELS=.*/ENABLE_DOCKER_MODELS=true/" .env
    else
        sed -i "s/ENABLE_DOCKER_MODELS=.*/ENABLE_DOCKER_MODELS=false/" .env
    fi
    
    log_success "Environment configuration updated"
}

# Function to select deployment profile
select_deployment_profile() {
    echo ""
    echo "🎯 Choose Deployment Profile:"
    echo ""
    echo "1. Hybrid (Recommended) - Docker Model Runner + Ollama fallback"
    echo "2. Enterprise - Docker Model Runner only (requires Docker Model Runner)"
    echo "3. Standard - Ollama only (traditional approach)"
    echo "4. Development - Minimal setup for testing"
    echo ""
    
    while true; do
        read -p "Select profile (1-4): " profile_choice
        case $profile_choice in
            1)
                PROFILE="hybrid"
                MODEL_STRATEGY="hybrid_docker_preferred"
                break
                ;;
            2)
                if [ "$DOCKER_MODELS_AVAILABLE" = "true" ]; then
                    PROFILE="enterprise"
                    MODEL_STRATEGY="docker_only"
                    break
                else
                    log_error "Docker Model Runner not available. Please choose another option."
                fi
                ;;
            3)
                PROFILE="ollama"
                MODEL_STRATEGY="ollama_only"
                break
                ;;
            4)
                PROFILE="ollama"
                MODEL_STRATEGY="ollama_only"
                DEPLOYMENT_ENV="development"
                break
                ;;
            *)
                echo "Invalid choice. Please enter 1-4."
                ;;
        esac
    done
    
    log_success "Selected profile: $PROFILE (strategy: $MODEL_STRATEGY)"
}

# Function to start services
start_services() {
    log_info "Starting Cortex Suite services..."
    
    # Build arguments
    BUILD_ARGS=""
    if [ "$DOCKER_MODELS_AVAILABLE" = "true" ]; then
        BUILD_ARGS="--build-arg ENABLE_DOCKER_MODELS=true"
    fi
    
    # Docker Compose command
    COMPOSE_CMD="docker compose -f $DOCKER_COMPOSE_FILE"
    
    # Add profile flags
    case $PROFILE in
        "hybrid")
            COMPOSE_CMD="$COMPOSE_CMD --profile hybrid"
            ;;
        "enterprise")
            COMPOSE_CMD="$COMPOSE_CMD --profile docker-models --profile enterprise"
            ;;
        "ollama")
            COMPOSE_CMD="$COMPOSE_CMD --profile ollama"
            ;;
    esac
    
    # Stop any existing services
    $COMPOSE_CMD down >/dev/null 2>&1 || true
    
    # Start services
    log_info "Building and starting services (this may take a few minutes)..."
    if ! $COMPOSE_CMD up -d --build; then
        log_error "Failed to start services!"
        echo "Check the logs with: $COMPOSE_CMD logs"
        exit 1
    fi
    
    log_success "Services started successfully"
}

# Function to wait for services
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for API
    log_info "Waiting for API server..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            log_success "API server is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            log_warning "API server seems slow to start"
        fi
        sleep 2
    done
    
    # Wait for UI
    log_info "Waiting for Streamlit UI..."
    for i in {1..20}; do
        if curl -s http://localhost:8501/_stcore/health >/dev/null 2>&1; then
            log_success "Streamlit UI is ready"
            break
        fi
        if [ $i -eq 20 ]; then
            log_warning "UI seems slow to start"
        fi
        sleep 2
    done
}

# Function to setup initial models
setup_models() {
    log_info "Setting up AI models..."
    
    echo ""
    echo "🤖 AI Model Setup:"
    echo ""
    echo "The system needs AI models to function. This process will:"
    echo "- Download required models (~11-15GB)"
    echo "- Take 10-30 minutes depending on internet speed"
    echo "- Continue in the background while you use the interface"
    echo ""
    
    read -p "Start model setup now? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        log_info "Skipping model setup - you can configure models later via the Setup Wizard"
        return
    fi
    
    # Start model initialization
    COMPOSE_CMD="docker compose -f $DOCKER_COMPOSE_FILE"
    
    case $PROFILE in
        "hybrid"|"enterprise")
            $COMPOSE_CMD --profile init up model-init-hybrid --no-deps
            ;;
        "ollama")
            $COMPOSE_CMD --profile init up model-init --no-deps
            ;;
    esac
    
    log_success "Model setup initiated - check the Setup Wizard for progress"
}

# Function to display final information
show_completion_info() {
    echo ""
    echo "🎉 Cortex Suite is now running!"
    echo ""
    echo "📱 Access Points:"
    echo "   Main Application: http://localhost:8501"
    echo "   API Documentation: http://localhost:8000/docs"
    echo "   Setup Wizard: http://localhost:8501/0_Setup_Wizard"
    echo ""
    echo "💡 Quick Start:"
    echo "   1. Visit the Setup Wizard to complete configuration"
    echo "   2. Upload documents via Knowledge Ingest"
    echo "   3. Try AI Research or Proposal Generation"
    echo ""
    echo "🔧 Management Commands:"
    echo "   Stop:      $COMPOSE_CMD down"
    echo "   Restart:   $COMPOSE_CMD restart"
    echo "   Logs:      $COMPOSE_CMD logs -f"
    echo "   Status:    $COMPOSE_CMD ps"
    echo ""
    echo "📊 System Info:"
    echo "   Profile:   $PROFILE"
    echo "   Strategy:  $MODEL_STRATEGY" 
    echo "   Environment: $DEPLOYMENT_ENV"
    echo "   Docker Models: $DOCKER_MODELS_AVAILABLE"
    echo ""
    
    # Check if Setup Wizard is needed
    if [ ! -f "$HOME/.cortex/setup_progress.json" ]; then
        log_info "First time setup detected - visit the Setup Wizard to complete configuration"
    fi
}

# Function to handle cleanup
cleanup() {
    echo ""
    log_info "Cleaning up..."
    # Add any cleanup tasks here
}

# Trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    echo "Starting Cortex Suite Hybrid Model Architecture..."
    echo ""
    
    # Check if we're in the right directory
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        log_error "Docker compose file not found: $DOCKER_COMPOSE_FILE"
        echo "Make sure you're running this script from the docker/ directory"
        exit 1
    fi
    
    # Run setup steps
    check_prerequisites
    check_disk_space
    select_deployment_profile
    setup_environment
    start_services
    wait_for_services
    setup_models
    show_completion_info
    
    echo ""
    log_success "Setup complete! 🚀"
    
    # Optional: Open browser
    read -p "Open browser to Cortex Suite? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        if command -v xdg-open >/dev/null; then
            xdg-open http://localhost:8501
        elif command -v open >/dev/null; then
            open http://localhost:8501
        else
            echo "Please open http://localhost:8501 in your browser"
        fi
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            MODEL_STRATEGY="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --env)
            DEPLOYMENT_ENV="$2"
            shift 2
            ;;
        --help)
            echo "Cortex Suite Hybrid Launcher"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --strategy STRATEGY    Model distribution strategy"
            echo "                         (hybrid_docker_preferred, docker_only, ollama_only)"
            echo "  --profile PROFILE      Deployment profile (hybrid, enterprise, ollama)"
            echo "  --env ENVIRONMENT      Deployment environment (development, production)"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Interactive setup"
            echo "  $0 --strategy hybrid_docker_preferred --profile hybrid"
            echo "  $0 --profile enterprise --env production"
            echo ""
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main