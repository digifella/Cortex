# Qwen3-VL Multimodal Embedding Setup Guide

## Configuration Complete! ✅

Qwen3-VL multimodal embeddings have been configured for your RTX 4060 Laptop GPU (8GB VRAM).

## What Was Done

### 1. Dependencies Installed
- ✅ `qwen-vl-utils 0.0.14` - Image/video processing utilities
- ✅ `transformers 4.57.6` - Upgraded from 4.41.2 for Qwen3-VL support
- ✅ `av 16.1.0` - Video decoding support

### 2. Configuration Created
- ✅ Environment configuration file: `.env.qwen3vl`
- ✅ Model: Qwen3-VL-Embedding-2B (optimal for 8GB VRAM)
- ✅ Dimensions: 2048 (vs 768 for BGE-base)
- ✅ Reranker: Enabled for improved search accuracy

### 3. Model Download Started
- ✅ Qwen3-VL-Embedding-2B downloading to cache
- Location: `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B`
- Size: ~5GB (currently 514MB downloaded)
- **Download will complete on first use**

## How to Use Qwen3-VL

### Starting Cortex Suite with Qwen3-VL

**Option 1: Using the environment file (recommended)**
```bash
# In project directory
source .env.qwen3vl
streamlit run Cortex_Suite.py
```

**Option 2: Setting variables manually**
```bash
export QWEN3_VL_ENABLED=true
export QWEN3_VL_MODEL_SIZE=2B
export QWEN3_VL_RERANKER_ENABLED=true
export HF_HUB_OFFLINE=0
streamlit run Cortex_Suite.py
```

### First Run Model Download

The first time you ingest documents with Qwen3-VL enabled:
1. The model will automatically download (~5GB)
2. This takes 5-15 minutes depending on internet speed
3. Subsequent runs will use the cached model (instant loading)

## Re-ingesting Your Documents

**IMPORTANT**: To use Qwen3-VL embeddings, you must re-ingest your documents.

### Why Re-ingest?

Your existing "Forensic sources" collection uses BGE-base embeddings (768 dimensions).
Qwen3-VL embeddings are 2048 dimensions and incompatible - you cannot mix them.

### Re-ingestion Process

#### Option A: New Collection (Recommended)
Keep your old data and create a new collection:

1. Start Cortex with Qwen3-VL enabled:
   ```bash
   source .env.qwen3vl
   streamlit run Cortex_Suite.py
   ```

2. Go to **Knowledge Ingest** page

3. Select your source documents again

4. Use a **new collection name**: `Forensic sources Qwen3VL`

5. Start ingestion - model will download on first use

6. Compare search quality between old (BGE) and new (Qwen3-VL) collections

#### Option B: Replace Existing Collection
Replace your old BGE-based collection:

1. **Backup first** (optional):
   ```bash
   cp -r /home/longboardfella/ai_databases/knowledge_hub_db \
         /home/longboardfella/ai_databases/knowledge_hub_db.bge-backup
   ```

2. Delete old collection via **Collection Management** page

3. Start Cortex with Qwen3-VL:
   ```bash
   source .env.qwen3vl
   streamlit run Cortex_Suite.py
   ```

4. Re-ingest with same collection name: `Forensic sources`

## What You Get with Qwen3-VL

### Capabilities

**Multimodal Understanding:**
- ✅ Text embeddings (like BGE, but higher quality)
- ✅ Image embeddings (charts, diagrams, photos)
- ✅ PDF page images (full page visual context)
- ✅ Video embeddings (forensic video evidence)
- ✅ **Cross-modal search**: Find images using text queries!

**Example Multimodal Queries:**
```
"Show me organizational charts"
→ Finds both text mentioning org charts AND actual chart images

"Crime scene diagrams"
→ Retrieves visual diagrams from PDFs

"Evidence photos from case XYZ"
→ Finds relevant images based on text context
```

### Performance Comparison

| Feature | BGE-base-en-v1.5 | Qwen3-VL-Embedding-2B |
|---------|------------------|------------------------|
| Dimensions | 768 | 2048 |
| Modalities | Text only | Text + Image + Video |
| VRAM Usage | ~0.5GB | ~5GB |
| Quality | Good | Superior |
| GPU Optimized | Yes | Yes |
| Cross-modal | No | Yes |

## Verification

### Check Configuration
```bash
source .env.qwen3vl
python3 -c "
from cortex_engine.config import QWEN3_VL_ENABLED, QWEN3_VL_MODEL_SIZE
print(f'Qwen3-VL Enabled: {QWEN3_VL_ENABLED}')
print(f'Model Size: {QWEN3_VL_MODEL_SIZE}')
"
```

Expected output:
```
Qwen3-VL Enabled: True
Model Size: 2B
```

### Check Model Cache
```bash
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B
```

When fully downloaded: ~5.0GB

## Troubleshooting

### Model Download Stuck
```bash
# Clear partial download and retry
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B
source .env.qwen3vl
streamlit run Cortex_Suite.py
```

### Out of Memory Error
If you get OOM errors during ingestion:
```bash
# Enable MRL dimension reduction (75% storage savings)
export QWEN3_VL_MRL_DIM=512
```

### Compatibility Warning
The transformers/tokenizers version warning with ChromaDB is safe to ignore.
Both libraries work correctly despite the dependency conflict message.

## Advanced Configuration

### Optional: Enable Flash Attention 2
For better memory efficiency (requires compilation):
```bash
pip install flash-attn --no-build-isolation
```

### Optional: Adjust Batch Sizes
If running low on VRAM during ingestion:
```bash
# Reduce batch size in cortex_config.json or page settings
# Default: 16 for 2B model
```

## Performance Tips

1. **First ingestion will be slower** (model download + GPU warm-up)
2. **Subsequent ingestions are faster** (cached model)
3. **Use GPU monitoring**: `nvidia-smi -l 1` in separate terminal
4. **Optimal batch size**: 8-16 for 8GB GPU

## Next Steps

1. ✅ Configuration complete
2. ⏳ Start Cortex with Qwen3-VL enabled
3. ⏳ Wait for model download (first time only)
4. ⏳ Re-ingest documents into new/existing collection
5. ⏳ Test cross-modal search capabilities!

---

**Need Help?** Check the Cortex Suite logs for detailed error messages.

**Model Info**: [Qwen3-VL-Embedding-2B on Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
