#!/bin/bash
set -e

echo "🚀 Starting Cortex Suite (CPU-Optimized)"
python -c "from cortex_engine.version_config import VERSION_DISPLAY; print('Version:', VERSION_DISPLAY)"
echo "📅 $(date)"
echo "💻 Platform: $(uname -m) ($(uname -s))"
echo "🐳 Docker Environment: $([ -f /.dockerenv ] && echo 'Yes' || echo 'No')"
echo "🔧 Multi-Platform Support: Intel x86_64, Apple Silicon M-series, ARM64 PCs"
echo ""

echo "🔧 Ensuring correct data permissions..."
# Prefer host-mounted volumes when present
PREFERRED_AI_DB="/data/ai_databases"
if [ -d "$PREFERRED_AI_DB" ]; then
    echo "📂 Detected external AI database mount at $PREFERRED_AI_DB"
    export AI_DATABASE_PATH="$PREFERRED_AI_DB"
else
    echo "📂 Using internal AI database path at $AI_DATABASE_PATH"
    mkdir -p "$AI_DATABASE_PATH"
fi

PREFERRED_SOURCE="/data/knowledge_base"
if [ -d "$PREFERRED_SOURCE" ]; then
    echo "📂 Detected external knowledge source mount at $PREFERRED_SOURCE"
    export KNOWLEDGE_SOURCE_PATH="$PREFERRED_SOURCE"
else
    echo "📂 Using internal knowledge source path at $KNOWLEDGE_SOURCE_PATH"
    mkdir -p "$KNOWLEDGE_SOURCE_PATH"
fi

if [ -w /data ]; then
    echo "Data directory writable, skipping permission fix"
else
    echo "Fixing data directory permissions..."
fi

echo "🤖 Starting Ollama service..."
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_ORIGINS=*
ollama serve &
OLLAMA_PID=$!

echo "⏳ Waiting for Ollama to initialize..."
for i in {1..60}; do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "⚠️ Ollama taking longer than expected, continuing..."
        echo "🔍 Ollama process status: $(ps aux | grep ollama || echo 'Not found')"
        break
    fi
    echo "   ... attempt $i/60"
    sleep 3
done

echo "🔗 Starting API server..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 &
API_PID=$!

echo "🖥️ Starting Streamlit UI..."
streamlit run Cortex_Suite.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false &
STREAMLIT_PID=$!

echo "⏳ Waiting for services to start..."
sleep 10

echo ""
echo "🎉 Cortex Suite is now accessible!"
echo "🌐 Access at: http://localhost:8501"
echo "🔗 API docs: http://localhost:8000/docs"
echo "📦 AI models will download in background..."
echo ""

{
    echo "📦 Starting AI model downloads..."
    if ! ollama list 2>/dev/null | grep -q "mistral:latest"; then
        echo "⬇️ Downloading Mistral model (4.4GB)..."
        ollama pull mistral:latest
        echo "✅ Mistral model ready!"
    fi
    
    if ! ollama list 2>/dev/null | grep -q "mistral-small3.2"; then
        echo "⬇️ Downloading Mistral Small model (15GB)..."
        ollama pull mistral-small3.2
        echo "✅ Mistral Small model ready!"
    fi
    
    echo "🎯 All AI models are now ready!"
    echo "🚀 Full functionality is now available at http://localhost:8501"
} &
MODEL_DOWNLOAD_PID=$!

cleanup() {
    echo ""
    echo "🛑 Shutting down Cortex Suite..."
    kill $OLLAMA_PID $API_PID $STREAMLIT_PID $MODEL_DOWNLOAD_PID 2>/dev/null || true
    echo "✅ Shutdown complete"
    exit 0
}

trap cleanup SIGTERM SIGINT

while true; do
    if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
        echo "❌ Streamlit process died, restarting..."
        streamlit run Cortex_Suite.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false &
        STREAMLIT_PID=$!
    fi
    
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "❌ API process died, restarting..."
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 &
        API_PID=$!
    fi
    
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "❌ Ollama process died, restarting..."
        ollama serve &
        OLLAMA_PID=$!
    fi
    
    sleep 30
done
