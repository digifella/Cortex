# Dockerfile for Cortex Suite - All-in-One Container
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    graphviz \
    graphviz-dev \
    gcc \
    g++ \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app user first
RUN useradd --create-home --shell /bin/bash cortex

# Install Ollama - download and extract from GitHub releases
# Release is now distributed as .tar.zst archive
RUN apt-get update && apt-get install -y zstd && rm -rf /var/lib/apt/lists/* && \
    curl -fsSL -o /tmp/ollama.tar.zst https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst && \
    mkdir -p /tmp/ollama && \
    tar --zstd -xf /tmp/ollama.tar.zst -C /tmp/ollama && \
    cp /tmp/ollama/bin/ollama /usr/local/bin/ollama && \
    chmod +x /usr/local/bin/ollama && \
    ln -sf /usr/local/bin/ollama /usr/bin/ollama && \
    rm -rf /tmp/ollama /tmp/ollama.tar.zst && \
    mkdir -p /home/cortex/.ollama && \
    chown -R cortex:cortex /home/cortex/.ollama && \
    ollama --version

USER cortex
WORKDIR /home/cortex/app

# Copy requirements first for better caching
COPY --chown=cortex:cortex requirements.txt .

# Install Python dependencies with cache for faster rebuilds
# Exclude spacy model (installed separately below for reliability)
# Increased timeout to 300 seconds for slow/unstable network connections
# Extract package names and versions only (no URLs, no hashes) for cross-platform compatibility
# Disable hash checking to avoid transient PyPI hash mismatch errors
RUN --mount=type=cache,target=/home/cortex/.cache/pip \
    grep -v "^en-core-web-sm" requirements.txt | \
    grep -v "^#" | grep -v "^$" | \
    sed 's/ @.*//' | sed 's/#.*//' > requirements-base.txt && \
    pip config set install.require-hashes false && \
    pip install --retries 5 --timeout 300 -r requirements-base.txt

# PyTorch installation is now handled in requirements.txt with CPU-only wheels
# This provides universal compatibility across x86_64, ARM64, and Snapdragon architectures
# For GPU acceleration, users can manually install CUDA wheels post-deployment

# Add user's local bin to PATH for installed packages
ENV PATH="/home/cortex/.local/bin:$PATH"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download required NLTK data for document processing
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Install Docling for enhanced document processing (optional)
# Note: Install with proper NumPy constraints to prevent conflicts
RUN pip install 'docling>=1.0.0' 'numpy>=1.26.4,<2.0.0' || echo "Docling installation failed, will use legacy readers"

# Pre-download embedding model for offline operation
# Temporarily disable offline mode during build to allow download
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')" || echo "WARNING: Embedding model download failed - container will require internet on first use"
# Re-enable offline mode for runtime
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy application code
COPY --chown=cortex:cortex . .

# Create data directories
RUN mkdir -p /home/cortex/data/ai_databases \
    /home/cortex/data/knowledge_base \
    /home/cortex/logs \
    /home/cortex/app/logs

# Set default environment variables
ENV AI_DATABASE_PATH=/home/cortex/data/ai_databases
ENV KNOWLEDGE_SOURCE_PATH=/home/cortex/data/knowledge_base
ENV GRAPHVIZ_DOT_EXECUTABLE=/usr/bin/dot
ENV LLM_PROVIDER=ollama
ENV OLLAMA_MODEL=mistral:latest
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_ORIGINS=*

# Expose ports
EXPOSE 8501 8000

# Create startup script with service-first architecture (heredoc for safer quoting)
RUN cat <<'EOF' > /home/cortex/app/start.sh
#!/bin/bash
set -e

echo "🚀 Starting Cortex Suite (CPU-Optimized)"
python -c 'from cortex_engine.version_config import VERSION_DISPLAY; print("Version:", VERSION_DISPLAY)'
echo "📅 $(date)"
echo "💻 Platform: $(uname -m) ($(uname -s))"
echo "🐳 Docker Environment: $([ -f /.dockerenv ] && echo "Yes" || echo "No")"
echo "🔧 Multi-Platform Support: Intel x86_64, Apple Silicon M-series, ARM64 PCs"
echo ""

# Fix data directory permissions (in case of root-owned volumes)
echo "🔧 Ensuring correct data permissions..."
if [ -w /data ]; then
    echo "Data directory writable, skipping permission fix"
else
    echo "Fixing data directory permissions..."
fi

# Start Ollama service in background
echo "🤖 Starting Ollama service..."
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_ORIGINS=*
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready (longer timeout)
echo "⏳ Waiting for Ollama to initialize..."
for i in {1..60}; do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "⚠️ Ollama taking longer than expected, continuing..."
        echo "🔍 Ollama process status: $(ps aux | grep ollama || echo "Not found")"
        break
    fi
    echo "   ... attempt $i/60"
    sleep 3

done

# Start API server immediately (will handle missing models gracefully)
echo "🔗 Starting API server..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 &
API_PID=$!

# Start Streamlit app immediately (with setup progress page)
echo "🖥️ Starting Streamlit UI..."
streamlit run Cortex_Suite.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false &
STREAMLIT_PID=$!

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

echo ""
echo "🎉 Cortex Suite (CPU-Optimized) is now accessible!"
echo "🌐 Access at: http://localhost:8501"
echo "🔗 API docs: http://localhost:8000/docs"
echo "🤖 Ollama: ENABLED"
echo ""

# Download models in background (lightweight set for CPU)
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
EOF

RUN chmod +x /home/cortex/app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health && curl -f http://localhost:8000/health

CMD ["/home/cortex/app/start.sh"]
