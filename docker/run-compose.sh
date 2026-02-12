#!/bin/bash
# Cortex Suite - Docker Compose Launcher
# Supports both portable (containerized) and external storage modes
#
# Usage:
#   ./run-compose.sh              # Portable mode (default)
#   ./run-compose.sh --external   # External storage mode
#   ./run-compose.sh --gpu        # Force GPU profile (fail-fast if unavailable)
#   ./run-compose.sh --cpu        # Force CPU profile
#   ./run-compose.sh --stop       # Stop all services
#   ./run-compose.sh --down       # Stop and remove containers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

get_env_from_dotenv() {
    local key="$1"
    local val
    val=$(grep -E "^${key}=" .env 2>/dev/null | tail -n1 | cut -d'=' -f2- | tr -d '"' | tr -d "'" || true)
    echo "$val"
}

is_port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$port$"
        return $?
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"$port" -sTCP:LISTEN -Pn >/dev/null 2>&1
        return $?
    fi
    if command -v netstat >/dev/null 2>&1; then
        netstat -an 2>/dev/null | grep -Eq "[\\.:]$port[[:space:]]+.*LISTEN"
        return $?
    fi
    return 1
}

find_next_free_port() {
    local start_port="$1"
    local label="$2"
    local candidate=$((start_port + 1))
    local max_port=$((start_port + 200))
    while [ "$candidate" -le "$max_port" ]; do
        if ! is_port_in_use "$candidate"; then
            echo "$candidate"
            return 0
        fi
        candidate=$((candidate + 1))
    done
    echo -e "${RED}Could not find free port for $label near $start_port.${NC}" >&2
    return 1
}

echo ""
echo "==============================================="
echo "   CORTEX SUITE - Docker Compose Launcher"
echo "   Date: $(date)"
echo "==============================================="
echo ""

# Parse arguments
STORAGE_MODE="portable"
ACTION="start"
EXECUTION_MODE="auto"

while [[ $# -gt 0 ]]; do
    case $1 in
        --external|-e)
            STORAGE_MODE="external"
            shift
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        --down)
            ACTION="down"
            shift
            ;;
        --gpu)
            EXECUTION_MODE="gpu"
            shift
            ;;
        --cpu)
            EXECUTION_MODE="cpu"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --external, -e    Use external storage (host filesystem)"
            echo "  --gpu             Force GPU profile (fail-fast if unavailable)"
            echo "  --cpu             Force CPU profile"
            echo "  --stop            Stop all services"
            echo "  --down            Stop and remove containers"
            echo "  --help, -h        Show this help"
            echo ""
            echo "Storage Modes:"
            echo "  Portable (default): Data stored in Docker volumes - fully transportable"
            echo "  External:           Data stored on host filesystem - easy access/backup"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Docker is not running!${NC}"
    echo "Please start Docker Desktop and try again."
    exit 1
fi
echo -e "${GREEN}Docker is running${NC}"

# Handle stop/down actions
if [ "$ACTION" = "stop" ]; then
    echo -e "${YELLOW}Stopping Cortex Suite services...${NC}"
    docker compose stop
    echo -e "${GREEN}Services stopped${NC}"
    exit 0
fi

if [ "$ACTION" = "down" ]; then
    echo -e "${YELLOW}Stopping and removing Cortex Suite containers...${NC}"
    docker compose down
    echo -e "${GREEN}Containers removed${NC}"
    exit 0
fi

# Check/create .env file
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${BLUE}Created .env configuration file${NC}"
    else
        echo -e "${RED}.env.example not found!${NC}"
        exit 1
    fi
fi

# External storage mode - validate paths
if [ "$STORAGE_MODE" = "external" ]; then
    echo ""
    echo "==============================================="
    echo "   EXTERNAL STORAGE MODE"
    echo "==============================================="
    echo ""

    # Check if external paths are configured
    EXTERNAL_AI_PATH=$(grep -E '^EXTERNAL_AI_DATABASE_PATH=' .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'" || true)
    EXTERNAL_KB_PATH=$(grep -E '^EXTERNAL_KNOWLEDGE_PATH=' .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'" || true)

    if [ -z "$EXTERNAL_AI_PATH" ]; then
        echo -e "${YELLOW}EXTERNAL_AI_DATABASE_PATH not set in .env${NC}"
        echo ""
        read -p "Enter path for AI databases (e.g., /mnt/f/ai_databases): " EXTERNAL_AI_PATH

        if [ -z "$EXTERNAL_AI_PATH" ]; then
            echo -e "${RED}Path required for external storage mode${NC}"
            exit 1
        fi

        # Create directory if needed
        if [ ! -d "$EXTERNAL_AI_PATH" ]; then
            echo "Creating directory: $EXTERNAL_AI_PATH"
            mkdir -p "$EXTERNAL_AI_PATH"
        fi

        # Save to .env
        echo "EXTERNAL_AI_DATABASE_PATH=$EXTERNAL_AI_PATH" >> .env
    fi

    if [ -z "$EXTERNAL_KB_PATH" ]; then
        echo ""
        read -p "Enter path for knowledge source (e.g., /mnt/e/KB_Test) [optional]: " EXTERNAL_KB_PATH

        if [ -n "$EXTERNAL_KB_PATH" ]; then
            if [ ! -d "$EXTERNAL_KB_PATH" ]; then
                echo "Creating directory: $EXTERNAL_KB_PATH"
                mkdir -p "$EXTERNAL_KB_PATH"
            fi
            echo "EXTERNAL_KNOWLEDGE_PATH=$EXTERNAL_KB_PATH" >> .env
        fi
    fi

    echo ""
    echo -e "${GREEN}External storage configuration:${NC}"
    echo "  AI Database: $EXTERNAL_AI_PATH"
    [ -n "$EXTERNAL_KB_PATH" ] && echo "  Knowledge Base: $EXTERNAL_KB_PATH"
    echo ""

    COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.external.yml"
else
    echo ""
    echo "==============================================="
    echo "   PORTABLE STORAGE MODE (Default)"
    echo "==============================================="
    echo ""
    echo "Data will be stored in Docker volumes:"
    echo "  - cortex_ai_databases (vector store, knowledge graph)"
    echo "  - cortex_knowledge_base (source documents)"
    echo ""
    echo "To backup: docker run --rm -v cortex_ai_databases:/data -v \$(pwd):/backup alpine tar czf /backup/backup.tar.gz /data"
    echo ""

    COMPOSE_CMD="docker compose"
fi

# Resolve host ports (defaults can be overridden in .env).
UI_PORT="$(get_env_from_dotenv CORTEX_UI_PORT)"; UI_PORT="${UI_PORT:-8501}"
API_PORT="$(get_env_from_dotenv CORTEX_API_PORT)"; API_PORT="${API_PORT:-8000}"
OLLAMA_PORT="$(get_env_from_dotenv CORTEX_OLLAMA_PORT)"; OLLAMA_PORT="${OLLAMA_PORT:-11434}"
export CORTEX_UI_PORT="$UI_PORT"
export CORTEX_API_PORT="$API_PORT"
export CORTEX_OLLAMA_PORT="$OLLAMA_PORT"

# Resolve CPU/GPU profile with explicit runtime checks.
has_gpu_cli=false
has_gpu_runtime=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    has_gpu_cli=true
fi
if docker info 2>/dev/null | grep -qi "nvidia"; then
    has_gpu_runtime=true
fi

if [ "$EXECUTION_MODE" = "cpu" ]; then
    echo -e "${BLUE}CPU mode forced by --cpu${NC}"
    PROFILE="--profile cpu"
elif [ "$EXECUTION_MODE" = "gpu" ]; then
    echo "GPU mode requested (--gpu). Validating NVIDIA runtime..."
    if [ "$has_gpu_cli" != "true" ]; then
        echo -e "${RED}GPU mode failed: nvidia-smi not available or GPU not detected.${NC}"
        echo "Install NVIDIA drivers (or run with --cpu)."
        exit 1
    fi
    if [ "$has_gpu_runtime" != "true" ]; then
        echo -e "${RED}GPU mode failed: Docker NVIDIA runtime is not detected.${NC}"
        echo "Install/configure NVIDIA Container Toolkit, then retry (or run with --cpu)."
        exit 1
    fi
    echo -e "${GREEN}NVIDIA GPU + Docker runtime detected${NC}"
    PROFILE="--profile gpu"
else
    echo "Checking for NVIDIA GPU..."
    if [ "$has_gpu_cli" = "true" ] && [ "$has_gpu_runtime" = "true" ]; then
        echo -e "${GREEN}NVIDIA GPU + Docker runtime detected${NC}"
        PROFILE="--profile gpu"
    else
        if [ "$has_gpu_cli" = "true" ] && [ "$has_gpu_runtime" != "true" ]; then
            echo -e "${YELLOW}NVIDIA GPU detected but Docker NVIDIA runtime missing; falling back to CPU.${NC}"
            echo -e "${YELLOW}Tip: install NVIDIA Container Toolkit or use --gpu after setup.${NC}"
        else
            echo -e "${BLUE}No NVIDIA GPU detected - using CPU mode${NC}"
        fi
        PROFILE="--profile cpu"
    fi
fi

# Stop opposite profile container if present to avoid Cortex-to-Cortex port collisions.
if [ "$PROFILE" = "--profile gpu" ]; then
    docker compose --profile cpu stop cortex-suite-cpu >/dev/null 2>&1 || true
else
    docker compose --profile gpu stop cortex-suite-gpu >/dev/null 2>&1 || true
fi

# Host port preflight checks.
for pair in "$UI_PORT:Streamlit UI" "$API_PORT:API" "$OLLAMA_PORT:Ollama"; do
    p="${pair%%:*}"
    label="${pair#*:}"
    if is_port_in_use "$p"; then
        next_port="$(find_next_free_port "$p" "$label")" || exit 1
        echo -e "${YELLOW}$label port $p is busy; using $next_port instead.${NC}"
        if [ "$label" = "Streamlit UI" ]; then
            UI_PORT="$next_port"
            export CORTEX_UI_PORT="$UI_PORT"
        elif [ "$label" = "API" ]; then
            API_PORT="$next_port"
            export CORTEX_API_PORT="$API_PORT"
        else
            OLLAMA_PORT="$next_port"
            export CORTEX_OLLAMA_PORT="$OLLAMA_PORT"
        fi
    fi
done

# Build and start services
echo ""
echo "Starting Cortex Suite services..."
echo "Command: $COMPOSE_CMD $PROFILE up -d --build"
echo "Host ports: UI=$CORTEX_UI_PORT API=$CORTEX_API_PORT OLLAMA=$CORTEX_OLLAMA_PORT"
echo ""

$COMPOSE_CMD $PROFILE up -d --build

echo ""
echo "Waiting for services to start..."
sleep 10

# Check service health
echo ""
echo "Checking service status..."
$COMPOSE_CMD ps

echo ""
echo "==============================================="
echo -e "${GREEN}   Cortex Suite is starting!${NC}"
echo "==============================================="
echo ""
echo "Access your Cortex Suite at:"
echo "  Main Application: http://localhost:$CORTEX_UI_PORT"
echo "  API Documentation: http://localhost:$CORTEX_API_PORT/docs"
echo ""
echo "Useful commands:"
echo "  View logs:     docker compose logs -f"
echo "  Stop:          ./run-compose.sh --stop"
echo "  Remove:        ./run-compose.sh --down"
echo ""
if [ "$STORAGE_MODE" = "portable" ]; then
    echo "Storage: PORTABLE (Docker volumes)"
    echo "  Backup:  docker run --rm -v cortex_ai_databases:/data -v \$(pwd):/backup alpine tar czf /backup/cortex_backup.tar.gz /data"
else
    echo "Storage: EXTERNAL ($EXTERNAL_AI_PATH)"
fi
echo ""
