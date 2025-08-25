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

# Install Ollama and give cortex user access
RUN curl -fsSL https://ollama.ai/install.sh | sh && \
    chown -R cortex:cortex /usr/local/bin/ollama /usr/bin/ollama 2>/dev/null || true && \
    mkdir -p /home/cortex/.ollama && \
    chown -R cortex:cortex /home/cortex/.ollama

USER cortex
WORKDIR /home/cortex/app

# Copy requirements first for better caching
COPY --chown=cortex:cortex requirements.txt .

# Install Python dependencies (architecture-neutral)
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with intelligent platform detection for universal compatibility
# Supports Intel x86_64, Apple Silicon M-series, ARM64 PCs
RUN ARCH=$(uname -m) && \
    echo "🔍 Detected architecture: $ARCH" && \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        echo "🍎 ARM64 architecture detected (Apple Silicon M-series/ARM64 PC)"; \
        echo "📦 Installing PyTorch CPU-optimized for ARM64"; \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    elif [ "$ARCH" = "x86_64" ] && nvidia-smi > /dev/null 2>&1; then \
        echo "🚀 Intel x86_64 with NVIDIA GPU detected"; \
        echo "⚡ Installing PyTorch with CUDA acceleration"; \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    else \
        echo "💻 Intel x86_64 CPU-only detected"; \
        echo "📦 Installing PyTorch CPU-optimized"; \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Add user's local bin to PATH for installed packages
ENV PATH="/home/cortex/.local/bin:$PATH"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download required NLTK data for document processing
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Install Docling for enhanced document processing (optional)
# Note: Install with proper NumPy constraints to prevent conflicts
RUN pip install 'docling>=1.0.0' 'numpy>=1.26.4,<2.0.0' || echo "Docling installation failed, will use legacy readers"

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
ENV OLLAMA_MODEL=mistral:7b-instruct-v0.3-q4_K_M
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_ORIGINS=*

# Expose ports
EXPOSE 8501 8000

# Create startup script with service-first architecture
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 Starting Cortex Suite v2.1.0 (Universal Architecture)"\n\
echo "📅 $(date)"\n\
echo "💻 Platform: $(uname -m) ($(uname -s))"\n\
echo "🐳 Docker Environment: $([ -f /.dockerenv ] && echo "Yes" || echo "No")"\n\
echo "🔧 Multi-Platform Support: Intel x86_64, Apple Silicon M-series, ARM64 PCs"\n\
echo "📋 Docker Ingestion Fixes: Applied (2025-08-24)"\n\
echo ""\n\
\n\
# Start Ollama service in background\n\
echo "🤖 Starting Ollama service..."\n\
export OLLAMA_HOST=0.0.0.0:11434\n\
export OLLAMA_ORIGINS=*\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
# Wait for Ollama to be ready (longer timeout)\n\
echo "⏳ Waiting for Ollama to initialize..."\n\
for i in {1..60}; do\n\
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then\n\
        echo "✅ Ollama is ready!"\n\
        break\n\
    fi\n\
    if [ $i -eq 60 ]; then\n\
        echo "⚠️ Ollama taking longer than expected, continuing..."\n\
        echo "🔍 Ollama process status: $(ps aux | grep ollama || echo 'Not found')"\n\
        break\n\
    fi\n\
    echo "   ... attempt $i/60"\n\
    sleep 3\n\
done\n\
\n\
# Start API server immediately (will handle missing models gracefully)\n\
echo "🔗 Starting API server..."\n\
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 &\n\
API_PID=$!\n\
\n\
# Start Streamlit app immediately (with setup progress page)\n\
echo "🖥️ Starting Streamlit UI..."\n\
streamlit run Cortex_Suite.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false &\n\
STREAMLIT_PID=$!\n\
\n\
# Wait for services to be ready\n\
echo "⏳ Waiting for services to start..."\n\
sleep 10\n\
\n\
echo ""\n\
echo "🎉 Cortex Suite is now accessible!"\n\
echo "🌐 Access at: http://localhost:8501"\n\
echo "🔗 API docs: http://localhost:8000/docs"\n\
echo "📦 AI models will download in background..."\n\
echo ""\n\
\n\
# Download models in background (non-blocking)\n\
{\n\
    echo "📦 Starting AI model downloads..."\n\
    if ! ollama list 2>/dev/null | grep -q "mistral:7b-instruct-v0.3-q4_K_M"; then\n\
        echo "⬇️ Downloading Mistral model (4.4GB)..."\n\
        ollama pull mistral:7b-instruct-v0.3-q4_K_M\n\
        echo "✅ Mistral model ready!"\n\
    fi\n\
    \n\
    if ! ollama list 2>/dev/null | grep -q "mistral-small3.2"; then\n\
        echo "⬇️ Downloading Mistral Small model (15GB)..."\n\
        ollama pull mistral-small3.2\n\
        echo "✅ Mistral Small model ready!"\n\
    fi\n\
    \n\
    echo "🎯 All AI models are now ready!"\n\
    echo "🚀 Full functionality is now available at http://localhost:8501"\n\
} &\n\
MODEL_DOWNLOAD_PID=$!\n\
\n\
# Function to handle shutdown\n\
cleanup() {\n\
    echo ""\n\
    echo "🛑 Shutting down Cortex Suite..."\n\
    kill $OLLAMA_PID $API_PID $STREAMLIT_PID $MODEL_DOWNLOAD_PID 2>/dev/null || true\n\
    echo "✅ Shutdown complete"\n\
    exit 0\n\
}\n\
\n\
# Set up signal handlers\n\
trap cleanup SIGTERM SIGINT\n\
\n\
# Keep container running and monitor processes\n\
while true; do\n\
    # Check if critical processes are still running\n\
    if ! kill -0 $STREAMLIT_PID 2>/dev/null; then\n\
        echo "❌ Streamlit process died, restarting..."\n\
        streamlit run Cortex_Suite.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false &\n\
        STREAMLIT_PID=$!\n\
    fi\n\
    \n\
    if ! kill -0 $API_PID 2>/dev/null; then\n\
        echo "❌ API process died, restarting..."\n\
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 &\n\
        API_PID=$!\n\
    fi\n\
    \n\
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then\n\
        echo "❌ Ollama process died, restarting..."\n\
        ollama serve &\n\
        OLLAMA_PID=$!\n\
    fi\n\
    \n\
    sleep 30\n\
done' > /home/cortex/app/start.sh

RUN chmod +x /home/cortex/app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health && curl -f http://localhost:8000/health

CMD ["/home/cortex/app/start.sh"]