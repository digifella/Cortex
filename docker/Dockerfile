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

# Install Python dependencies with cache for faster rebuilds
# Exclude spacy model (installed separately below for reliability)
# Increased timeout to 300 seconds for slow/unstable network connections
# Extract package names and versions only (no URLs, no hashes) for cross-platform compatibility
RUN --mount=type=cache,target=/home/cortex/.cache/pip \
    grep -v "^en-core-web-sm" requirements.txt | \
    grep -v "^#" | grep -v "^$" | \
    sed 's/ @.*//' | sed 's/#.*//' > requirements-base.txt && \
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

# Copy startup script with stable quoting
COPY --chown=cortex:cortex start_cpu.sh /home/cortex/app/start.sh
RUN chmod +x /home/cortex/app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health && curl -f http://localhost:8000/health

CMD ["/home/cortex/app/start.sh"]
