#!/bin/bash
# Start Cortex Suite with Qwen3-VL Multimodal Embeddings
# RTX 4060 Laptop GPU (8GB VRAM) - Optimized Configuration

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🚀 Starting Cortex Suite with Qwen3-VL Embeddings  "
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Load Qwen3-VL configuration
source .env.qwen3vl

echo "📋 Configuration:"
echo "   • Model: Qwen3-VL-Embedding-2B"
echo "   • Dimensions: 2048"
echo "   • Reranker: Enabled"
echo "   • GPU: RTX 4060 Laptop (8GB VRAM)"
echo ""

# Check if model is already downloaded
MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B"
if [ -d "$MODEL_CACHE" ]; then
    MODEL_SIZE=$(du -sh "$MODEL_CACHE" | cut -f1)
    echo "✅ Model cached: $MODEL_SIZE"
else
    echo "⏳ Model will download on first use (~5GB, takes 5-15 min)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Starting Streamlit...                              "
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start Streamlit
streamlit run Cortex_Suite.py
