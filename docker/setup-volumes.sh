#!/bin/bash

# ## File: docker/setup-volumes.sh
# Version: 1.0.0
# Date: 2025-08-08
# Purpose: Interactive script to configure Docker volume mounts for Cortex Suite

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
FLEXIBLE_COMPOSE="$SCRIPT_DIR/docker-compose.flexible.yml"

print_header() {
    echo -e "${BLUE}"
    echo "═══════════════════════════════════════════════════════"
    echo "  Cortex Suite Docker Volume Configuration"
    echo "═══════════════════════════════════════════════════════"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

check_requirements() {
    print_step "Checking requirements..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    print_info "Requirements satisfied ✓"
}

backup_existing_compose() {
    if [ -f "$COMPOSE_FILE" ]; then
        print_step "Backing up existing docker-compose.yml..."
        cp "$COMPOSE_FILE" "$COMPOSE_FILE.backup.$(date +%Y%m%d_%H%M%S)"
        print_info "Backup created"
    fi
}

detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM=Linux;;
        Darwin*)    PLATFORM=Mac;;
        CYGWIN*|MINGW*|MSYS*) PLATFORM=Windows;;
        *)          PLATFORM="Unknown";;
    esac
    
    print_info "Detected platform: $PLATFORM"
}

suggest_paths() {
    print_step "Suggesting common document paths for $PLATFORM:"
    echo
    
    case $PLATFORM in
        Linux)
            echo "  Common Linux paths:"
            echo "    📁 $HOME/Documents"
            echo "    📁 $HOME/Downloads" 
            echo "    📁 /media/*/  (external drives)"
            echo "    📁 /mnt/*/    (mounted filesystems)"
            if grep -q Microsoft /proc/version 2>/dev/null; then
                echo "  WSL2 Windows paths:"
                echo "    📁 /mnt/c/Users/$(whoami)/Documents"
                echo "    📁 /mnt/c/Users/$(whoami)/Desktop"
            fi
            ;;
        Mac)
            echo "  Common macOS paths:"
            echo "    📁 $HOME/Documents"
            echo "    📁 $HOME/Downloads"
            echo "    📁 $HOME/Desktop"
            echo "    📁 /Volumes/*  (external drives)"
            ;;
        Windows)
            echo "  Common Windows paths:"
            echo "    📁 C:/Users/$(whoami)/Documents"
            echo "    📁 C:/Users/$(whoami)/Downloads"
            echo "    📁 D:/  (additional drives)"
            ;;
    esac
    echo
}

validate_path() {
    local path="$1"
    
    if [ ! -d "$path" ]; then
        print_warning "Path does not exist: $path"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
    fi
    
    if [ ! -r "$path" ]; then
        print_warning "Path is not readable: $path"
        return 1
    fi
    
    return 0
}

configure_mounts() {
    print_step "Configuring volume mounts..."
    echo
    
    MOUNTS=()
    ENV_PATHS=()
    
    while true; do
        echo "Enter a path to mount (or press Enter to finish):"
        read -r HOST_PATH
        
        if [ -z "$HOST_PATH" ]; then
            break
        fi
        
        # Expand tilde
        HOST_PATH="${HOST_PATH/#\~/$HOME}"
        
        if ! validate_path "$HOST_PATH"; then
            continue
        fi
        
        echo "Enter container mount point (e.g., documents, projects):"
        read -r MOUNT_NAME
        
        if [ -z "$MOUNT_NAME" ]; then
            print_warning "Mount name cannot be empty"
            continue
        fi
        
        # Sanitize mount name
        MOUNT_NAME=$(echo "$MOUNT_NAME" | sed 's/[^a-zA-Z0-9_-]//g')
        
        echo "Mount as read-only? (Y/n):"
        read -n 1 -r READONLY
        echo
        
        if [[ $READONLY =~ ^[Nn]$ ]]; then
            PERM="rw"
        else
            PERM="ro"
        fi
        
        CONTAINER_PATH="/host/$MOUNT_NAME"
        MOUNT_LINE="      - $HOST_PATH:$CONTAINER_PATH:$PERM"
        
        MOUNTS+=("$MOUNT_LINE")
        ENV_PATHS+=("$CONTAINER_PATH")
        
        print_info "Added mount: $HOST_PATH → $CONTAINER_PATH ($PERM)"
        echo
    done
    
    if [ ${#MOUNTS[@]} -eq 0 ]; then
        print_warning "No mounts configured. Using default internal volumes only."
        return
    fi
    
    # Create the new compose file
    print_step "Creating docker-compose.yml with configured mounts..."
    
    cp "$FLEXIBLE_COMPOSE" "$COMPOSE_FILE"
    
    # Insert volume mounts into the compose file
    for mount in "${MOUNTS[@]}"; do
        # Add to both cortex-api and cortex-ui services
        sed -i "/# - \/path\/to\/your\/documents:\/host\/documents:ro/a\\$mount" "$COMPOSE_FILE"
    done
    
    # Update environment variable with available paths
    ENV_PATHS_STR=$(IFS=,; echo "${ENV_PATHS[*]}")
    sed -i "s|HOST_MOUNT_PATHS=/host/documents,/host/projects,/host/home,/host/external,/host/network|HOST_MOUNT_PATHS=$ENV_PATHS_STR|g" "$COMPOSE_FILE"
    
    print_info "Docker Compose configuration updated"
}

verify_configuration() {
    print_step "Verifying configuration..."
    
    if ! docker-compose -f "$COMPOSE_FILE" config > /dev/null 2>&1; then
        print_error "Generated docker-compose.yml is invalid"
        return 1
    fi
    
    print_info "Configuration is valid ✓"
    
    echo
    print_info "Configured volume mounts:"
    grep -A 20 "volumes:" "$COMPOSE_FILE" | grep "host/" | head -10
}

restart_services() {
    echo
    read -p "Restart Cortex Suite services now? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Restarting services..."
        
        cd "$SCRIPT_DIR"
        docker-compose down
        docker-compose up -d
        
        print_info "Services restarted with new volume configuration"
        echo
        print_info "Access Cortex Suite at: http://localhost:8501"
        print_info "Your mounted directories are available at /host/* paths"
    else
        print_info "To apply changes, run:"
        print_info "  cd $SCRIPT_DIR"
        print_info "  docker-compose down"
        print_info "  docker-compose up -d"
    fi
}

main() {
    print_header
    
    check_requirements
    detect_platform
    backup_existing_compose
    suggest_paths
    configure_mounts
    verify_configuration
    restart_services
    
    echo
    print_step "Setup complete!"
    print_info "See docker/VOLUME_MOUNTING_GUIDE.md for more advanced configurations"
}

# Handle interrupts gracefully
trap 'echo; print_warning "Setup interrupted"; exit 1' INT TERM

# Run main function
main "$@"