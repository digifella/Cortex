#!/bin/bash
# Start Cortex Suite with BGE embedding model forced
# This ensures the stable BGE model is used instead of auto-detected NVIDIA model

echo "============================================================"
echo "Starting Cortex Suite with BGE Embedding Model"
echo "============================================================"
echo ""

# Force BGE model via environment variable
export CORTEX_EMBED_MODEL="BAAI/bge-base-en-v1.5"

echo "✅ Environment variable set: CORTEX_EMBED_MODEL=${CORTEX_EMBED_MODEL}"
echo ""

# Verify the setting
echo "Verifying model configuration..."
python3 << 'EOF'
import os
os.environ["CORTEX_EMBED_MODEL"] = "BAAI/bge-base-en-v1.5"

from cortex_engine.config import EMBED_MODEL
print(f"✅ Config will load: {EMBED_MODEL}")

if EMBED_MODEL == "BAAI/bge-base-en-v1.5":
    print("✅ BGE model correctly configured!")
else:
    print(f"❌ ERROR: Expected BGE but got {EMBED_MODEL}")
    exit(1)
EOF

echo ""
echo "============================================================"
echo "Starting Streamlit..."
echo "============================================================"
echo ""

# Start Streamlit with the environment variable active
streamlit run Cortex_Suite.py
