#!/bin/bash
# Docker Distribution Sync Utility
# Version: v1.0.1
# Date: 2025-08-26
# Ensures docker subdirectory stays in sync with main project

echo "🔄 Syncing Docker Distribution..."

# Core application files
echo "📋 Syncing core files..."
cp Cortex_Suite.py docker/
cp requirements.txt docker/
cp .env.example docker/

# API directory
echo "🔌 Syncing API..."
rsync -av --delete api/ docker/api/

# Core engine
echo "⚙️ Syncing cortex_engine..."
rsync -av --delete cortex_engine/ docker/cortex_engine/

# Pages directory
echo "📄 Syncing pages..."
rsync -av --delete pages/ docker/pages/

# Scripts directory
echo "🔧 Syncing scripts..."
rsync -av --delete scripts/ docker/scripts/

# Check for version increments in changed files
echo "🔍 Checking version consistency..."

# Function to check version format
check_version_format() {
    local file="$1"
    local pattern="$2"
    
    if [[ -f "$file" ]]; then
        if grep -q "$pattern" "$file"; then
            version=$(grep "$pattern" "$file" | head -1)
            echo "  ✅ $file: $version"
        else
            echo "  ⚠️  $file: No version found with pattern '$pattern'"
        fi
    fi
}

# Check main pages
echo "📊 Version audit:"
check_version_format "docker/pages/3_Knowledge_Search.py" "PAGE_VERSION = "
check_version_format "docker/Cortex_Suite.py" "APP_VERSION = "
check_version_format "docker/run-cortex.sh" "CORTEX SUITE v"

echo "✅ Docker distribution sync complete!"
echo "📦 Ready for distribution testing."