# Adaptive Embedding Selection Guide

## Overview

Cortex Suite now features **fully adaptive embedding selection** that automatically chooses the best embedding model for ANY hardware configuration - from laptop GPUs to data center servers, Docker containers to bare metal.

**Zero configuration required** - it just works! ✨

## How It Works

The system automatically detects:
1. **GPU presence** (NVIDIA CUDA support)
2. **VRAM available** (2GB, 6GB, 8GB, 16GB, 40GB+ tiers)
3. **Dependencies installed** (qwen-vl-utils for multimodal support)
4. **Environment** (Docker, WSL, bare metal)

Based on this, it selects the optimal embedding approach:

```
┌─────────────────────────────────────────────────────────────┐
│           ADAPTIVE EMBEDDING DECISION TREE                  │
└─────────────────────────────────────────────────────────────┘

                    Hardware Detection
                            │
                            ▼
                ┌───────────────────────┐
                │   NVIDIA GPU Found?   │
                └───────────┬───────────┘
                           Yes│        │No
                    ┌──────────┘        └──────────┐
                    ▼                              ▼
        ┌──────────────────────┐       ┌──────────────────┐
        │  Check VRAM & Deps   │       │   BGE-base       │
        └───────────┬──────────┘       │   (CPU-friendly) │
                    │                  │   768D, text-only│
                    ▼                  └──────────────────┘
    ┌───────────────────────────┐
    │ qwen-vl-utils installed?  │
    └────────┬──────────────────┘
             │
    ┌────────┴────────┐
    Yes│             │No
       ▼              ▼
 ┌─────────┐    ┌──────────────┐
 │ VRAM?   │    │  NV-Embed-v2 │
 └────┬────┘    │ (GPU-optimized)│
      │         │ 4096D, text-only│
      ▼         └──────────────┘
 ┌──────────────────────────────┐
 │  6GB+: Qwen3-VL-2B/8B       │
 │  (Multimodal: text+image+video)│
 │  2048D/4096D                │
 └──────────────────────────────┘
```

## Automatic Selection Matrix

| Hardware | Dependencies | Selected Approach | Model | Dimensions | Multimodal |
|----------|--------------|-------------------|-------|------------|------------|
| **No GPU** | - | BGE-base | BAAI/bge-base-en-v1.5 | 768 | ❌ |
| **NVIDIA 2-6GB** | - | NV-Embed-v2 | nvidia/NV-Embed-v2 | 4096 | ❌ |
| **NVIDIA 6-10GB** | qwen-vl-utils ✅ | Qwen3-VL-2B | Qwen/Qwen3-VL-Embedding-2B | 2048 | ✅ |
| **NVIDIA 10-16GB** | qwen-vl-utils ✅ | Qwen3-VL-2B | Qwen/Qwen3-VL-Embedding-2B | 2048 | ✅ |
| **NVIDIA 16-24GB** | qwen-vl-utils ✅ | Qwen3-VL-8B | Qwen/Qwen3-VL-Embedding-8B | 4096 | ✅ |
| **NVIDIA 24-40GB** | qwen-vl-utils ✅ | Qwen3-VL-8B | Qwen/Qwen3-VL-Embedding-8B | 4096 | ✅ |
| **NVIDIA 40GB+** | qwen-vl-utils ✅ | Qwen3-VL-8B | Qwen/Qwen3-VL-Embedding-8B | 4096 | ✅ |

## Real-World Examples

### Example 1: Your Laptop (RTX 4060 8GB)
```
Hardware: RTX 4060 Laptop GPU (8GB VRAM)
Dependencies: qwen-vl-utils installed ✅
→ Auto-selects: Qwen3-VL-Embedding-2B
→ Multimodal: Yes (text + images + video)
→ Dimensions: 2048
```

### Example 2: Docker on Server (RTX 4090 24GB)
```
Hardware: RTX 4090 (24GB VRAM)
Dependencies: qwen-vl-utils in container ✅
→ Auto-selects: Qwen3-VL-Embedding-8B
→ Multimodal: Yes
→ Dimensions: 4096
→ Reranker: Qwen3-VL-Reranker-2B
```

### Example 3: Cloud VM (No GPU)
```
Hardware: CPU only
Dependencies: N/A
→ Auto-selects: BGE-base
→ Multimodal: No (text-only)
→ Dimensions: 768
```

### Example 4: Older GPU (GTX 1660 6GB)
```
Hardware: GTX 1660 (6GB VRAM)
Dependencies: qwen-vl-utils not installed ❌
→ Auto-selects: NV-Embed-v2
→ Multimodal: No (text-only)
→ Dimensions: 4096
```

## Manual Overrides

You can override the automatic selection if needed:

### Force Specific Model
```bash
# Use a specific model regardless of hardware
export CORTEX_EMBED_MODEL="nvidia/NV-Embed-v2"
streamlit run Cortex_Suite.py
```

### Force Qwen3-VL On
```bash
# Force Qwen3-VL even if auto-selection would choose something else
export QWEN3_VL_ENABLED=true
export QWEN3_VL_MODEL_SIZE=2B  # or 8B or auto
streamlit run Cortex_Suite.py
```

### Force Qwen3-VL Off
```bash
# Disable Qwen3-VL even if hardware supports it
export QWEN3_VL_ENABLED=false
streamlit run Cortex_Suite.py
```

## Docker Compatibility

The adaptive selection is **fully Docker-aware**:

### Auto-Detection in Docker
```yaml
# docker-compose.yml
services:
  cortex:
    image: cortex-suite:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    # NO CONFIGURATION NEEDED!
    # System auto-detects GPU and selects optimal model
```

### Manual Configuration in Docker
```yaml
# docker-compose.yml
services:
  cortex:
    image: cortex-suite:latest
    environment:
      # Optional: Force specific configuration
      - QWEN3_VL_ENABLED=true
      - QWEN3_VL_MODEL_SIZE=auto
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

## Verification Commands

### Check What's Auto-Selected
```bash
python3 -c "
from cortex_engine.config import get_embedding_strategy
strategy = get_embedding_strategy()
print(f'Approach: {strategy[\"approach\"]}')
print(f'Model: {strategy[\"model\"]}')
print(f'Dimensions: {strategy[\"dimensions\"]}')
print(f'Multimodal: {strategy[\"multimodal\"]}')
print(f'Reason: {strategy[\"reason\"]}')
"
```

### Check GPU Detection
```bash
python3 -c "
from cortex_engine.utils.smart_model_selector import detect_nvidia_gpu
has_gpu, info = detect_nvidia_gpu()
print(f'GPU: {has_gpu}')
if has_gpu:
    print(f'Name: {info[\"device_name\"]}')
    print(f'VRAM: {info.get(\"memory_total_gb\", 0):.1f}GB')
"
```

## Migration from Manual Configuration

### Old Way (Manual)
```bash
# Required manual configuration
export QWEN3_VL_ENABLED=true
export QWEN3_VL_MODEL_SIZE=2B
export HF_HUB_OFFLINE=0
source .env.qwen3vl
streamlit run Cortex_Suite.py
```

### New Way (Automatic)
```bash
# Just start - it auto-detects everything!
streamlit run Cortex_Suite.py
```

**Your existing `.env.qwen3vl` file still works for manual override if needed!**

## Multi-Machine Flexibility

The adaptive approach makes it trivial to run on different machines:

### Same Codebase, Different Hardware

**Laptop (8GB):**
```bash
git clone https://github.com/youruser/cortex
cd cortex
streamlit run Cortex_Suite.py
# → Auto-selects Qwen3-VL-2B
```

**Workstation (24GB):**
```bash
git clone https://github.com/youruser/cortex
cd cortex
streamlit run Cortex_Suite.py
# → Auto-selects Qwen3-VL-8B
```

**Server (CPU only):**
```bash
git clone https://github.com/youruser/cortex
cd cortex
streamlit run Cortex_Suite.py
# → Auto-selects BGE-base
```

**No configuration changes needed!**

## Performance Characteristics

| Approach | VRAM | Speed | Quality | Multimodal | Use Case |
|----------|------|-------|---------|------------|----------|
| **Qwen3-VL-8B** | 16GB | Fast | Excellent | ✅ | High-end workstations |
| **Qwen3-VL-2B** | 5GB | Fast | Very Good | ✅ | Mid-range GPUs |
| **NV-Embed-v2** | 1.2GB | Very Fast | Excellent | ❌ | Text-only, GPU-optimized |
| **BGE-base** | 0.5GB | Medium | Good | ❌ | CPU systems, fallback |

## Troubleshooting

### Issue: Wrong Model Selected

**Check what was selected:**
```bash
python3 -c "from cortex_engine.config import get_embedding_strategy; print(get_embedding_strategy())"
```

**Force a different selection:**
```bash
export CORTEX_EMBED_MODEL="your-preferred-model"
# or
export QWEN3_VL_ENABLED=true  # for multimodal
```

### Issue: Qwen3-VL Not Auto-Selected

**Likely causes:**
1. qwen-vl-utils not installed: `pip install qwen-vl-utils`
2. VRAM < 6GB (get NV-Embed-v2 instead)
3. Manual override set: `unset QWEN3_VL_ENABLED`

### Issue: GPU Not Detected

**Check PyTorch CUDA:**
```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**If False, reinstall PyTorch with CUDA:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Architecture Benefits

### ✅ Zero Configuration
- Works out of the box on any hardware
- No manual model selection needed
- No environment variable setup required

### ✅ Intelligent Fallbacks
- Gracefully handles missing dependencies
- Automatically downgrades if VRAM insufficient
- CPU fallback if no GPU available

### ✅ Docker-Friendly
- Auto-detects Docker environment
- Works with GPU pass-through
- No special Docker configuration needed

### ✅ Portable
- Same codebase works on laptop, workstation, server
- Adapts to available resources automatically
- Easy deployment across different hardware

### ✅ Override-Friendly
- Manual overrides still work for power users
- Environment variables respected
- Backward compatible with old configs

## Summary

The adaptive embedding selection makes Cortex Suite truly **hardware-agnostic**:

- **On your laptop**: Uses Qwen3-VL-2B for multimodal search
- **On a server**: Uses Qwen3-VL-8B for maximum quality
- **In Docker**: Auto-detects GPU and configures appropriately
- **On CPU-only**: Falls back to BGE-base gracefully

**Just clone and run - it works everywhere!** 🚀

---

**Next Steps:**
1. Remove any manual configuration (optional, still works)
2. Just start Cortex: `streamlit run Cortex_Suite.py`
3. Check what was selected: See verification commands above
4. Enjoy optimal performance on any hardware!
