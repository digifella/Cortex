# GPU Support in Docker - Cortex Suite

## Overview

Cortex Suite now automatically detects and uses NVIDIA GPUs when running in Docker. This provides significant performance improvements for:

- **Embedding generation:** 2-5x faster (adaptive batch sizing up to 128)
- **Image processing:** GPU-accelerated VLM inference
- **Overall ingestion:** 40-60% faster with GPU acceleration

## Requirements

### 1. NVIDIA GPU
- CUDA compute capability 6.0+ (Pascal architecture or newer)
- Minimum 4GB VRAM (8GB+ recommended)
- Supported cards: GTX 1060+, RTX 20/30/40 series, Tesla, A100, etc.

### 2. NVIDIA Drivers
- **Windows:** NVIDIA Game Ready or Studio Driver 450.80.02+
- **Linux:** NVIDIA Driver 450.80.02+
- Check version: `nvidia-smi`

### 3. Docker Desktop with GPU Support
- **Windows:** Docker Desktop 4.19+ with WSL2 backend
- **Linux:** Docker 19.03+ with NVIDIA Container Toolkit

## Installation

### Windows Setup

1. **Install NVIDIA Drivers**
   ```bash
   # Check current driver
   nvidia-smi

   # Should show driver version and GPU info
   ```

2. **Enable WSL2 GPU Support**
   - Docker Desktop → Settings → General
   - Check "Use the WSL 2 based engine"
   - Docker Desktop → Settings → Resources → WSL Integration
   - Enable integration with your WSL2 distro

3. **Verify GPU Access**
   ```bash
   # Test Docker can see GPU
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

   # Should display your GPU info
   ```

### Linux Setup

1. **Install NVIDIA Container Toolkit**
   ```bash
   # Add NVIDIA package repositories
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   # Install nvidia-container-toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit

   # Restart Docker
   sudo systemctl restart docker
   ```

2. **Verify Installation**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

## Usage

### Automatic GPU Detection

The `run-cortex.bat` script automatically detects your GPU and builds the appropriate image:

```bash
# Run the installer
./run-cortex.bat

# Output will show:
#   DEBUG: Checking for NVIDIA GPU...
#   OK: NVIDIA GPU detected - Building GPU-accelerated image
#   BUILD: Using Dockerfile.gpu for CUDA support
```

### Manual GPU Control

To force CPU-only mode even with GPU available:
```bash
docker build -t cortex-suite -f Dockerfile .
docker run -d --name cortex-suite -p 8501:8501 -p 8000:8000 ...
```

To force GPU mode:
```bash
docker build -t cortex-suite -f Dockerfile.gpu .
docker run -d --name cortex-suite --gpus all -p 8501:8501 -p 8000:8000 ...
```

## Verification

### Check GPU Usage in Container

1. **Access running container:**
   ```bash
   docker exec -it cortex-suite bash
   ```

2. **Verify PyTorch sees GPU:**
   ```python
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
   ```

3. **Monitor GPU during ingestion:**
   ```bash
   # In another terminal
   nvidia-smi -l 1  # Updates every second
   ```

### Performance Dashboard

Access the Performance Dashboard in Cortex:
- Navigate to **Maintenance** → **Performance** tab
- Check **Device & GPU Information** section
- Should show:
  - Device: Your GPU name (e.g., "NVIDIA RTX 3080")
  - Type: CUDA GPU
  - Recommended Batch Size: Based on your GPU memory

## Troubleshooting

### GPU Not Detected

**Symptom:** Container shows "Using CPU for embeddings"

**Solutions:**
1. Verify nvidia-smi works on host
2. Check Docker has GPU access: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
3. Rebuild container: `docker rm cortex-suite && ./run-cortex.bat`

### Out of Memory Errors

**Symptom:** CUDA out of memory during ingestion

**Solutions:**
1. The system uses adaptive batch sizing, but you can manually reduce:
   - Check current batch size in Performance Dashboard
   - Typical sizes: 4GB GPU=16, 8GB=32, 16GB=64, 24GB=128
   - Contact support if issues persist

### Slow Performance

**Symptom:** GPU shows low utilization in nvidia-smi

**Possible causes:**
1. Small batch size (check Performance Dashboard)
2. CPU bottleneck in preprocessing
3. Disk I/O bottleneck
4. Check Performance Dashboard for detailed metrics

## Architecture Details

### GPU-Enabled Components

1. **Embedding Service** (`cortex_engine/embedding_service.py`)
   - Auto-detects CUDA/MPS/CPU
   - Adaptive batch sizing (4-128)
   - GPU memory monitoring

2. **Image Processing** (`cortex_engine/query_cortex.py`)
   - Parallel processing with ThreadPoolExecutor
   - GPU-accelerated VLM inference

3. **Performance Monitoring** (`cortex_engine/utils/gpu_monitor.py`)
   - Real-time GPU memory tracking
   - Batch size optimization
   - Device recommendations

### Docker Images

- **Dockerfile** - CPU-only (universal compatibility)
- **Dockerfile.gpu** - CUDA 12.1 enabled (NVIDIA GPUs)
- **requirements.txt** - Base dependencies
- **requirements-gpu.txt** - CUDA-enabled PyTorch

## Performance Benchmarks

Typical performance improvements with GPU acceleration:

| Operation | CPU (8 cores) | GPU (RTX 3080) | Speedup |
|-----------|---------------|----------------|---------|
| Embedding (100 docs) | ~15s | ~3s | **5x** |
| Image processing (10 images) | ~120s | ~40s | **3x** |
| Full ingestion (50 mixed files) | ~180s | ~60s | **3x** |
| Query with cache miss | ~2s | ~0.5s | **4x** |

*Actual performance varies by GPU model, document type, and system configuration.*

## Support

For GPU-related issues:
1. Check this guide first
2. Review logs: `docker logs cortex-suite`
3. Open GitHub issue with:
   - GPU model (`nvidia-smi` output)
   - Docker version (`docker --version`)
   - Error messages from logs
