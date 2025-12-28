#!/bin/bash
# Setup script to configure BGE embedding model for Cortex Suite
# This forces the system to use the stable BAAI/bge-base-en-v1.5 model
# instead of auto-detecting NVIDIA GPU and using nvidia/NV-Embed-v2

echo "============================================================"
echo "Cortex Suite - BGE Embedding Model Setup"
echo "============================================================"
echo ""

# Set environment variable for current session
export CORTEX_EMBED_MODEL="BAAI/bge-base-en-v1.5"
echo "✅ Set CORTEX_EMBED_MODEL=BAAI/bge-base-en-v1.5 for this session"
echo ""

# Add to .bashrc for persistence
BASHRC="$HOME/.bashrc"
ENV_LINE='export CORTEX_EMBED_MODEL="BAAI/bge-base-en-v1.5"'

if grep -q "CORTEX_EMBED_MODEL" "$BASHRC"; then
    echo "⚠️  CORTEX_EMBED_MODEL already exists in ~/.bashrc"
    echo "   Please verify it's set to: BAAI/bge-base-en-v1.5"
else
    echo "" >> "$BASHRC"
    echo "# Cortex Suite - Force BGE embedding model (stable)" >> "$BASHRC"
    echo "$ENV_LINE" >> "$BASHRC"
    echo "✅ Added CORTEX_EMBED_MODEL to ~/.bashrc for persistence"
fi

echo ""
echo "============================================================"
echo "Testing BGE Model..."
echo "============================================================"
echo ""

# Test the model works
python3 << 'EOF'
import os
os.environ["CORTEX_EMBED_MODEL"] = "BAAI/bge-base-en-v1.5"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from sentence_transformers import SentenceTransformer

print("Loading BGE model...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
test_emb = model.encode("test")
print(f"✅ BGE model works perfectly - dimension: {len(test_emb)}D")
EOF

echo ""
echo "============================================================"
echo "✅ BGE Model Setup Complete!"
echo "============================================================"
echo ""
echo "Next Steps:"
echo ""
echo "1. Clean up orphaned documents:"
echo "   - Open Cortex Suite → Maintenance page"
echo "   - Go to 'Recovery Tools' section"
echo "   - Click 'Clean Up Orphaned Documents'"
echo ""
echo "2. Verify embedding model in Maintenance page:"
echo "   - Should show: BAAI/bge-base-en-v1.5 (768D)"
echo "   - Status should be 'Compatible'"
echo ""
echo "3. Run fresh ingestion:"
echo "   - Go to Knowledge Ingest page"
echo "   - Select your 2092 document directory"
echo "   - Should complete in ~1-2 hours"
echo ""
echo "4. Start Cortex Suite:"
echo "   streamlit run Cortex_Suite.py"
echo ""
echo "============================================================"
echo ""
echo "🔒 Model locked to BGE for all future sessions"
echo "   To change: edit ~/.bashrc and remove CORTEX_EMBED_MODEL line"
echo ""
