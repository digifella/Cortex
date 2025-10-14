#!/bin/bash
# Cortex Suite - Mac Diagnostic Script
# Run this if the container builds but the app won't load

echo "========================================"
echo "Cortex Suite - Mac Diagnostics"
echo "========================================"
echo ""

echo "1. Checking if Docker is running..."
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running!"
    exit 1
fi
echo "✅ Docker is running"
echo ""

echo "2. Checking if cortex-suite container exists..."
if ! docker ps -a --format '{{.Names}}' | grep -q "^cortex-suite$"; then
    echo "❌ Container 'cortex-suite' not found!"
    echo "Please run ./run-cortex.sh first"
    exit 1
fi
echo "✅ Container exists"
echo ""

echo "3. Checking container status..."
if docker ps --format '{{.Names}}' | grep -q "^cortex-suite$"; then
    echo "✅ Container is RUNNING"
else
    echo "⚠️  Container is STOPPED"
    echo "Attempting to start..."
    docker start cortex-suite
    sleep 5
fi
echo ""

echo "4. Checking container logs (last 50 lines)..."
echo "----------------------------------------"
docker logs --tail 50 cortex-suite
echo "----------------------------------------"
echo ""

echo "5. Checking if processes are running inside container..."
echo "----------------------------------------"
docker exec cortex-suite ps aux | grep -E "streamlit|ollama|uvicorn|python" || echo "No processes found!"
echo "----------------------------------------"
echo ""

echo "6. Checking port bindings..."
echo "----------------------------------------"
docker port cortex-suite
echo "----------------------------------------"
echo ""

echo "7. Testing connectivity to services..."
if curl -s http://localhost:8501/_stcore/health >/dev/null 2>&1; then
    echo "✅ Streamlit (port 8501) is responding"
else
    echo "❌ Streamlit (port 8501) is NOT responding"
fi

if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ API (port 8000) is responding"
else
    echo "❌ API (port 8000) is NOT responding"
fi
echo ""

echo "8. Checking volume mounts..."
echo "----------------------------------------"
docker inspect cortex-suite --format='{{range .Mounts}}{{.Type}}: {{.Source}} -> {{.Destination}}{{println}}{{end}}'
echo "----------------------------------------"
echo ""

echo "9. Checking for common Mac issues..."

# Check if volumes have correct permissions
echo "Checking /data permissions inside container..."
docker exec cortex-suite ls -la /data 2>/dev/null | head -5 || echo "⚠️  Cannot access /data directory"

# Check if Streamlit is actually installed
echo "Checking if Streamlit is installed..."
docker exec cortex-suite which streamlit >/dev/null 2>&1 && echo "✅ Streamlit found" || echo "❌ Streamlit not found"

# Check if Python can import required modules
echo "Checking Python imports..."
docker exec cortex-suite python -c "import streamlit; print('✅ Streamlit import OK')" 2>/dev/null || echo "❌ Streamlit import FAILED"
echo ""

echo "========================================"
echo "Diagnostic Complete"
echo "========================================"
echo ""
echo "📋 Common Solutions:"
echo ""
echo "If container is running but services aren't responding:"
echo "  1. Check container logs above for errors"
echo "  2. Try: docker restart cortex-suite"
echo "  3. Check Docker Desktop Settings > Resources > File Sharing"
echo "     (ensure mounted directories are allowed)"
echo ""
echo "If you see permission errors:"
echo "  1. Run: docker exec -u root cortex-suite chown -R cortex:cortex /data"
echo "  2. Restart container: docker restart cortex-suite"
echo ""
echo "If ports are already in use:"
echo "  1. Stop: docker stop cortex-suite"
echo "  2. Remove: docker rm cortex-suite"
echo "  3. Re-run: ./run-cortex.sh"
echo ""
echo "For more help, share the logs above with support"
