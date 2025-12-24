# GPU Setup Guide for Cortex Suite

This guide helps you enable NVIDIA GPU acceleration for faster embeddings and processing.

## Quick Status Check

Open **Knowledge Ingest** page in Cortex Suite and check the status bar at the top:

- **🎮 GPU Detected**: Your GPU is recognized
- **💻 CPU Mode**: No GPU detected or not configured

## System Requirements

### Hardware
- NVIDIA GPU (any model with CUDA support)
- Minimum 4GB GPU memory (8GB+ recommended)

### Software
- **NVIDIA Drivers**: Latest drivers installed on host system
- **CUDA Toolkit**: Not required (PyTorch includes necessary CUDA runtime)
- **Python**: 3.11 (as used in Cortex Suite)

---

## Installation Instructions

### Windows (Native)

1. **Install NVIDIA Drivers**
   - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
   - Restart after installation

2. **Install PyTorch with CUDA**
   ```bash
   # Activate your virtual environment first
   venv\Scripts\activate

   # Uninstall CPU-only PyTorch
   pip uninstall torch torchvision torchaudio -y

   # Install CUDA-enabled PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify Installation**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   ```

### Linux

1. **Install NVIDIA Drivers**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install nvidia-driver-535

   # Verify
   nvidia-smi
   ```

2. **Install PyTorch with CUDA**
   ```bash
   # Activate virtual environment
   source venv/bin/activate

   # Uninstall CPU-only PyTorch
   pip uninstall torch torchvision torchaudio -y

   # Install CUDA-enabled PyTorch
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify Installation**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   ```

### WSL2 (Windows Subsystem for Linux) ⭐

**Special Instructions for WSL + NVIDIA GPU**

1. **Install NVIDIA Drivers on Windows Host**
   - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
   - **Do NOT install drivers inside WSL** - use Windows drivers only
   - Restart Windows after installation

2. **Verify GPU Visibility in WSL**
   ```bash
   # This should show your GPU
   /mnt/c/Windows/System32/nvidia-smi.exe
   ```

   Expected output:
   ```
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.x    |
   |-------------------------------+----------------------+----------------------+
   | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   +-------------------------------+----------------------+----------------------+
   |   0  Quadro RTX 8000    WDDM  | 00000000:01:00.0 Off |                  Off |
   ```

3. **Install PyTorch with CUDA in WSL**
   ```bash
   # Activate virtual environment
   source venv/bin/activate

   # Uninstall CPU-only PyTorch
   pip uninstall torch torchvision torchaudio -y

   # Install CUDA-enabled PyTorch (CUDA 12.1)
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Verify Installation in WSL**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   ```

   Expected output:
   ```
   CUDA available: True
   GPU: Quadro RTX 8000
   ```

5. **Restart Streamlit**
   ```bash
   # Stop current Streamlit (Ctrl+C)
   # Start again
   streamlit run Cortex_Suite.py
   ```

---

## Model Configuration

Once GPU is enabled, configure embedding models in Cortex Suite:

### In Knowledge Ingest Page

1. **Check Status Bar (Top)**
   - Should show: `🎮 [Your GPU Name]`
   - Shows current embedding model

2. **Open Sidebar Configuration**
   - GPU Status section shows your GPU with memory
   - Embedding Model section for selection

3. **Select NVIDIA Nemotron Model**
   - Model: `nvidia/NV-Embed-v2 ⭐ (Recommended)`
   - Quality: ⭐⭐⭐⭐⭐
   - Size: 1.2 GB
   - Best for: NVIDIA GPU systems

4. **Apply Changes**
   - Click "🔄 Apply Change"
   - Model will be downloaded on first use (~1.2GB)
   - Future runs use cached model

---

## Available Embedding Models

| Model | Quality | Speed | Size | Best For |
|-------|---------|-------|------|----------|
| **nvidia/NV-Embed-v2** ⭐ | ⭐⭐⭐⭐⭐ | ⚡⚡⚡⚡⚡ (GPU) | 1.2GB | NVIDIA GPUs |
| BAAI/bge-large-en-v1.5 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | 1.3GB | High-end CPUs |
| BAAI/bge-base-en-v1.5 | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | 0.5GB | Balanced (default) |
| BAAI/bge-small-en-v1.5 | ⭐⭐⭐ | ⚡⚡⚡⚡⚡ | 0.13GB | Low-resource systems |
| all-MiniLM-L6-v2 | ⭐⭐⭐ | ⚡⚡⚡⚡⚡ | 0.09GB | Very limited resources |

**Recommendation**: Use **nvidia/NV-Embed-v2** on NVIDIA GPUs for best quality and speed.

---

## Troubleshooting

### GPU Not Detected

**Check Detection Status:**
1. Go to Knowledge Ingest page
2. Look at status bar - should show GPU or CPU mode
3. Open sidebar → GPU Status section
4. Expand "Detection Issues" if shown

**Common Issues:**

#### Issue: "PyTorch installed without CUDA support"
```bash
# Solution: Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Issue: "nvidia-smi not found"
- **Windows**: Install NVIDIA drivers
- **Linux**: Install nvidia-driver package
- **WSL**: Install drivers on Windows host (not in WSL)

#### Issue: "GPU access blocked by operating system" (WSL)
This is expected in WSL. The system uses Windows nvidia-smi instead:
```bash
# Verify GPU visibility
/mnt/c/Windows/System32/nvidia-smi.exe

# Then install PyTorch with CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Issue: "Failed to initialize NVML" (WSL)
This is normal in WSL2. PyTorch will still work with CUDA after installation.

### Model Download Fails

**Issue**: Can't download embedding models

**Solutions**:
1. **Check Internet Connection**: Models download from HuggingFace
2. **Use Pre-Download**: In sidebar, click "⬇️ Download Now" before ingestion
3. **Check Firewall**: Ensure access to `huggingface.co`
4. **Disk Space**: Models require 0.09GB to 1.3GB each

### Performance Issues

**GPU Detected but Slow:**
1. **Check GPU Utilization**:
   ```bash
   # Windows
   nvidia-smi

   # WSL
   /mnt/c/Windows/System32/nvidia-smi.exe
   ```

2. **Verify Model Location**: Should show CUDA device
   ```bash
   python -c "import torch; from sentence_transformers import SentenceTransformer; m = SentenceTransformer('nvidia/NV-Embed-v2'); print(f'Model device: {m.device}')"
   ```

3. **Expected Output**: `Model device: cuda:0`

**Memory Errors:**
- Reduce batch size in processing settings
- Use smaller embedding model (bge-small or MiniLM)
- Check GPU memory: `nvidia-smi`

---

## Docker Deployment

### GPU Support in Docker

1. **Install NVIDIA Container Toolkit**
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Update docker-compose.yml**
   ```yaml
   services:
     cortex-suite:
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: all
                 capabilities: [gpu]
   ```

3. **Build with CUDA PyTorch**
   Update Dockerfile:
   ```dockerfile
   # Install PyTorch with CUDA support
   RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Run Container**
   ```bash
   docker-compose up -d
   ```

---

## Performance Expectations

### With NVIDIA GPU (nvidia/NV-Embed-v2)
- **Embedding Speed**: 50-100 documents/minute (depends on GPU)
- **Quality**: Highest (state-of-the-art Nemotron model)
- **Memory Usage**: ~1.5-2GB GPU memory
- **Recommended For**: All NVIDIA GPU systems

### Without GPU (CPU Mode, BAAI/bge-base)
- **Embedding Speed**: 10-30 documents/minute (depends on CPU)
- **Quality**: High (proven BGE model)
- **Memory Usage**: ~0.5-1GB RAM
- **Recommended For**: Systems without NVIDIA GPU

---

## Additional Resources

- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **NVIDIA Drivers**: https://www.nvidia.com/Download/index.aspx
- **WSL2 GPU Support**: https://docs.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

---

## Support

If you encounter issues:

1. **Check GPU Detection**: Knowledge Ingest page → Status bar and sidebar
2. **Review Detection Issues**: Sidebar → GPU Status → "Detection Issues" expander
3. **Check Logs**: `logs/ingestion.log` for detailed error messages
4. **Verify PyTorch**: Run verification commands above

For additional help, check the GitHub repository or file an issue with:
- GPU model
- Operating system (Windows/Linux/WSL)
- PyTorch version: `python -c "import torch; print(torch.__version__)"`
- CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
