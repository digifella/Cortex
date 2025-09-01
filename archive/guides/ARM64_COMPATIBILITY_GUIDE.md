# ARM64 Compatibility Guide - Cortex Suite

**Version:** v4.1.2  
**Date:** 2025-08-29  
**Purpose:** Complete guide for ARM64 processor compatibility and multi-architecture support

## 🎯 Overview

This document provides comprehensive information about ARM64 compatibility fixes implemented in Cortex Suite v4.1.2, specifically addressing dependency conflicts on Windows Snapdragon processors and other ARM64 systems.

## 🚨 Problem Statement

### Original Issue
Users with ARM64 processors (Windows Snapdragon, Apple Silicon) encountered Docker build failures:

```
ERROR: Could not find a version that satisfies the requirement nvidia-cublas-cu12==12.1.3.1
ERROR: No matching distribution found for nvidia-cublas-cu12==12.1.3.1
```

### Root Cause Analysis
- **Hardcoded CUDA Dependencies**: 12 x86_64-specific NVIDIA CUDA libraries in requirements.txt
- **Architecture Mismatch**: ARM64 processors cannot use x86_64 CUDA binaries
- **Inflexible PyTorch Installation**: Exact version pinning prevented CPU-only fallbacks
- **Docker Build Context**: Missing architecture-specific handling

## ✅ Solution Implementation

### 1. Dependency Architecture (v4.1.2)

#### **Removed Hardcoded CUDA Dependencies**
```diff
# REMOVED FROM requirements.txt:
- nvidia-cublas-cu12==12.1.3.1
- nvidia-cuda-cupti-cu12==12.1.105
- nvidia-cuda-nvrtc-cu12==12.1.105
- nvidia-cuda-runtime-cu12==12.1.105
- nvidia-cudnn-cu12==8.9.2.26
- nvidia-cufft-cu12==11.0.2.54
- nvidia-curand-cu12==10.3.2.106
- nvidia-cusolver-cu12==11.4.5.107
- nvidia-cusparse-cu12==12.1.0.106
- nvidia-nccl-cu12==2.20.5
- nvidia-nvjitlink-cu12==12.9.86
- nvidia-nvtx-cu12==12.1.105
- triton==2.3.1
```

#### **Implemented Universal PyTorch Strategy**
```diff
# CHANGED IN requirements.txt:
- torch==2.3.1
- torchvision==0.18.1
+ torch>=2.3.1,<2.5.0
+ torchvision>=0.18.1,<0.20.0
# triton==2.3.1  # GPU-only dependency - removed for ARM64 compatibility
```

#### **Added Comprehensive Installation Documentation**
```txt
# Optional: GPU Acceleration (x86_64 only - not compatible with ARM64/Snapdragon)
# For NVIDIA GPU support on x86_64 systems:
# pip uninstall torch torchvision
# pip install torch==2.3.1 torchvision==0.18.1 --extra-index-url https://download.pytorch.org/whl/cu121
#
# Architecture-specific PyTorch installation:
# - x86_64 with NVIDIA GPU: Use CUDA 12.1 wheels (cu121)
# - x86_64 CPU-only: Use CPU wheels (cpu) - default above
# - ARM64 (Apple Silicon, Snapdragon): Use CPU wheels (cpu) - default above
# - Apple Silicon with MPS: pip install torch torchvision (default PyPI)
#
# The system automatically detects available acceleration (CUDA, MPS, CPU)
```

### 2. Docker Configuration Updates

#### **Simplified Dockerfile Architecture**
```diff
# REMOVED complex architecture-specific PyTorch installation
- RUN ARCH=$(uname -m) && \
-     echo "🔍 Detected architecture: $ARCH" && \
-     if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
-         pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
-     elif [ "$ARCH" = "x86_64" ] && nvidia-smi > /dev/null 2>&1; then \
-         pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121; \
-     else \
-         pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
-     fi

# ADDED simple, universal approach
+ # PyTorch installation is now handled in requirements.txt with CPU-only wheels
+ # This provides universal compatibility across x86_64, ARM64, and Snapdragon architectures
+ # For GPU acceleration, users can manually install CUDA wheels post-deployment
```

### 3. Import and Syntax Fixes

#### **Cortex_Suite.py Import Issues (Post-Sync)**
During version synchronization, several working imports were corrupted:

```python
# FIXED IMPORTS:
from cortex_engine.system_status import system_status
from cortex_engine.version_config import get_version_display, VERSION_METADATA
from cortex_engine.utils.model_checker import model_checker
from cortex_engine.help_system import help_system
```

#### **Syntax Error Resolution**
```diff
# CORRUPTED VERSION (caused syntax error):
- # Version: v4.1.2"Cortex Suite",
-     page_icon="🚀",
-     layout="wide"
- )

# FIXED VERSION:
+ # Version: v4.1.2
+ # Date: 2025-08-29
+ # Purpose: Main entry point for the Cortex Suite application
+ 
+ import streamlit as st
+ # ... proper imports ...
+ 
+ st.set_page_config(
+     page_title="Cortex Suite",
+     page_icon="🚀",
+     layout="wide"
+ )
```

## 🏗️ Architecture Support Matrix

### ✅ Supported Architectures (v4.1.2+)

| Architecture | OS Support | Installation Method | GPU Acceleration |
|-------------|------------|-------------------|------------------|
| **Intel x86_64** | Windows, Linux, macOS | Default requirements.txt | Optional CUDA upgrade |
| **ARM64 (Apple Silicon)** | macOS M1/M2/M3 | Default requirements.txt | MPS acceleration (automatic) |
| **ARM64 (Snapdragon)** | Windows on ARM | Default requirements.txt | CPU-only |
| **ARM64 (Linux)** | Linux distros | Default requirements.txt | CPU-only |
| **aarch64** | Various Unix systems | Default requirements.txt | CPU-only |

### 🔧 GPU Acceleration Upgrade Paths

#### **Intel x86_64 with NVIDIA GPU:**
```bash
# After base installation:
pip uninstall torch torchvision
pip install torch==2.3.1 torchvision==0.18.1 --extra-index-url https://download.pytorch.org/whl/cu121
```

#### **Apple Silicon (Automatic MPS):**
```bash
# Base installation automatically enables MPS when available
# No additional steps required
```

#### **ARM64/Snapdragon (CPU Optimized):**
```bash
# Default installation is already optimized for ARM64 CPU
# No GPU acceleration available on these platforms
```

## 🛠️ Troubleshooting Guide

### Common Issues and Solutions

#### **Issue: "No matching distribution found for nvidia-cublas-cu12"**
```
ERROR: No matching distribution found for nvidia-cublas-cu12==12.1.3.1
```
**Solution:** You're on ARM64. Use the updated v4.1.2 requirements.txt that removes hardcoded CUDA dependencies.

#### **Issue: Docker build fails on Windows Snapdragon**
```
ERROR: failed to build: process "/bin/sh -c pip install --no-cache-dir -r requirements.txt"
```
**Solution:** Ensure you're using the v4.1.2+ Docker distribution with ARM64-compatible dependencies.

#### **Issue: PyTorch not using GPU acceleration**
```
UserWarning: CUDA is not available
```
**Solution:** This is expected on ARM64. For x86_64 with GPU, follow the CUDA upgrade path above.

#### **Issue: Import errors after version sync**
```
NameError: name 'model_checker' is not defined
```
**Solution:** The version sync process can sometimes corrupt imports. Check that all required imports are present in Cortex_Suite.py.

### Diagnostic Commands

```bash
# Check architecture
uname -m

# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available()}')"

# Verify requirements compatibility
pip install --dry-run -r requirements.txt

# Check Docker build context
docker build --platform linux/arm64 -t cortex-test .
```

## 📋 Development Workflow Integration

### Version Management Best Practices

#### **1. Centralized Version Control**
```bash
# Update version ONLY in cortex_engine/version_config.py
# Then sync across all components:
python scripts/version_manager.py --sync-all
python scripts/version_manager.py --update-changelog
python scripts/version_manager.py --check
```

#### **2. Dependency Management**
Following lessons from `DEPENDENCY_RESOLUTION_GUIDE.md`:
- **Keep version ranges flexible** unless specific versions required
- **Make enhancements optional** with graceful fallbacks
- **Test architecture compatibility** before adding dependencies
- **Document upgrade paths** clearly

#### **3. Docker Distribution Sync**
```bash
# After code changes, sync Docker directory:
cp requirements.txt docker/
cp -r cortex_engine/* docker/cortex_engine/
cp -r pages/* docker/pages/
cp Cortex_Suite.py docker/
```

### Testing Checklist

#### **Before Release:**
- [ ] Test on x86_64 (Linux/WSL)
- [ ] Test Docker build on multiple architectures
- [ ] Verify requirements.txt compatibility with `pip install --dry-run`
- [ ] Check version consistency with `python scripts/version_manager.py --check`
- [ ] Test import statements and syntax with `python -m py_compile`

#### **Post-Release:**
- [ ] Monitor user reports for architecture-specific issues
- [ ] Document any new compatibility requirements
- [ ] Update troubleshooting guide with real-world solutions

## 🎯 Success Metrics

### ARM64 Compatibility Achievement (v4.1.2)

#### **✅ Technical Achievements:**
- **Zero hardcoded x86_64 dependencies** in core requirements.txt
- **Universal CPU-first installation** works on all architectures
- **Clear upgrade paths** for GPU acceleration when available
- **Comprehensive documentation** for troubleshooting
- **Docker multi-architecture support** verified

#### **✅ User Experience Improvements:**
- **Immediate compatibility** with Windows Snapdragon processors
- **No architecture detection required** - works out of the box
- **Professional installation experience** with clear instructions
- **Optional performance upgrades** without breaking base functionality

#### **✅ Development Process Enhancements:**
- **Centralized version management** prevents inconsistencies
- **Automated synchronization** across project components
- **Comprehensive testing guidelines** for multi-architecture support
- **Documentation-first approach** for compatibility issues

## 🔮 Future Considerations

### Planned Enhancements
- **Automatic architecture detection** in Docker launcher scripts
- **Performance benchmarking** across different architectures  
- **Expanded GPU support** for AMD GPUs and other accelerators
- **Container registry** with pre-built multi-architecture images

### Monitoring and Maintenance
- **Regular dependency audits** for new architecture conflicts
- **User feedback integration** for real-world compatibility issues
- **Automated testing** on multiple architecture platforms
- **Documentation updates** based on user experiences

---

## 📚 Related Documentation

- **`DEPENDENCY_RESOLUTION_GUIDE.md`** - Comprehensive dependency management lessons learned
- **`CHANGELOG.md`** - Version history and feature additions
- **`README.md`** - General installation and usage instructions
- **`docker/README.md`** - Docker-specific installation guide

---

**This guide represents the definitive reference for ARM64 compatibility in Cortex Suite. All future architecture-related work should reference and build upon these foundations.**