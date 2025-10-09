"""
GPU Memory Monitoring and Batch Size Optimization
Automatically detects available GPU memory and suggests optimal batch sizes
for embedding generation to maximize throughput without OOM errors.
"""

from __future__ import annotations

import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# GPU Memory Information
# ============================================================================

@dataclass
class GPUMemoryInfo:
    """GPU memory statistics"""
    device_name: str
    total_memory_gb: float
    allocated_memory_gb: float
    reserved_memory_gb: float
    free_memory_gb: float
    utilization_percent: float
    is_cuda: bool
    is_mps: bool
    is_cpu: bool


def get_gpu_memory_info() -> GPUMemoryInfo:
    """
    Get current GPU memory statistics.

    Returns:
        GPUMemoryInfo with current memory state
    """
    # Check for CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_mem = torch.cuda.memory_reserved(0) / (1024**3)
        free_mem = total_mem - reserved_mem
        utilization = (allocated_mem / total_mem * 100) if total_mem > 0 else 0.0

        return GPUMemoryInfo(
            device_name=device_name,
            total_memory_gb=total_mem,
            allocated_memory_gb=allocated_mem,
            reserved_memory_gb=reserved_mem,
            free_memory_gb=free_mem,
            utilization_percent=utilization,
            is_cuda=True,
            is_mps=False,
            is_cpu=False
        )

    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't provide detailed memory info
        return GPUMemoryInfo(
            device_name="Apple Silicon GPU (MPS)",
            total_memory_gb=0.0,  # Not available via PyTorch
            allocated_memory_gb=0.0,
            reserved_memory_gb=0.0,
            free_memory_gb=0.0,
            utilization_percent=0.0,
            is_cuda=False,
            is_mps=True,
            is_cpu=False
        )

    # CPU fallback
    else:
        return GPUMemoryInfo(
            device_name="CPU",
            total_memory_gb=0.0,
            allocated_memory_gb=0.0,
            reserved_memory_gb=0.0,
            free_memory_gb=0.0,
            utilization_percent=0.0,
            is_cuda=False,
            is_mps=False,
            is_cpu=True
        )


# ============================================================================
# Batch Size Optimization
# ============================================================================

def get_optimal_batch_size(
    model_name: str = "BAAI/bge-base-en-v1.5",
    conservative: bool = True
) -> int:
    """
    Calculate optimal batch size based on available GPU memory.

    This function analyzes GPU memory and returns a safe batch size
    that maximizes throughput without risking OOM errors.

    Args:
        model_name: Embedding model name (for size estimation)
        conservative: Use conservative estimates (safer, default True)

    Returns:
        Recommended batch size (4 to 128)

    Batch Size Guidelines:
        - GPU with 24GB+ VRAM: 128 (maximum throughput)
        - GPU with 16GB VRAM: 64 (good balance)
        - GPU with 8-12GB VRAM: 32 (default, safe)
        - GPU with 4-8GB VRAM: 16 (conservative)
        - GPU with <4GB or MPS: 8 (minimal)
        - CPU: 4 (limited by CPU speed, not memory)
    """
    gpu_info = get_gpu_memory_info()

    # CPU - use small batch (CPU-bound, not memory-bound)
    if gpu_info.is_cpu:
        logger.info("💻 CPU mode: Using batch size 4 (CPU-optimized)")
        return 4

    # MPS (Apple Silicon) - use moderate batch
    if gpu_info.is_mps:
        logger.info("🍎 MPS mode: Using batch size 16 (Apple Silicon optimized)")
        return 16

    # CUDA - calculate based on available memory
    if gpu_info.is_cuda:
        total_memory = gpu_info.total_memory_gb
        free_memory = gpu_info.free_memory_gb

        # Safety margin (20% for conservative, 10% for aggressive)
        safety_margin = 0.20 if conservative else 0.10
        usable_memory = free_memory * (1 - safety_margin)

        # Estimate memory per sample (BGE-base uses ~100MB per 32 samples)
        # This is a rough estimate: 768 dimensions * 4 bytes * overhead
        memory_per_sample_gb = 0.003  # 3MB per sample (conservative estimate)

        # Calculate max batch size based on memory
        max_batch_size = int(usable_memory / memory_per_sample_gb)

        # Clamp to reasonable range with power-of-2 sizes
        if max_batch_size >= 128:
            batch_size = 128
            tier = "Maximum"
        elif max_batch_size >= 64:
            batch_size = 64
            tier = "High"
        elif max_batch_size >= 32:
            batch_size = 32
            tier = "Default"
        elif max_batch_size >= 16:
            batch_size = 16
            tier = "Conservative"
        elif max_batch_size >= 8:
            batch_size = 8
            tier = "Minimal"
        else:
            batch_size = 4
            tier = "Ultra-conservative"

        logger.info(
            f"🎯 GPU ({gpu_info.device_name}): {total_memory:.1f}GB total, "
            f"{free_memory:.1f}GB free → Batch size {batch_size} ({tier})"
        )

        return batch_size

    # Fallback
    return 16


def should_use_gpu_batching() -> bool:
    """
    Check if GPU batching would provide performance benefit.

    Returns:
        True if GPU is available and batching recommended
    """
    gpu_info = get_gpu_memory_info()
    return gpu_info.is_cuda or gpu_info.is_mps


def get_device_recommendations() -> Dict[str, any]:
    """
    Get comprehensive device and batch size recommendations.

    Returns:
        Dictionary with device info and recommendations
    """
    gpu_info = get_gpu_memory_info()
    optimal_batch = get_optimal_batch_size(conservative=True)
    aggressive_batch = get_optimal_batch_size(conservative=False)

    return {
        "device_info": {
            "name": gpu_info.device_name,
            "type": "cuda" if gpu_info.is_cuda else ("mps" if gpu_info.is_mps else "cpu"),
            "total_memory_gb": gpu_info.total_memory_gb,
            "free_memory_gb": gpu_info.free_memory_gb,
            "utilization_percent": gpu_info.utilization_percent
        },
        "batch_recommendations": {
            "conservative": optimal_batch,
            "aggressive": aggressive_batch,
            "recommended": optimal_batch,  # Default to conservative
        },
        "performance_tier": _get_performance_tier(optimal_batch),
        "use_gpu_batching": should_use_gpu_batching()
    }


def _get_performance_tier(batch_size: int) -> str:
    """Map batch size to performance tier."""
    if batch_size >= 128:
        return "Maximum (24GB+ GPU)"
    elif batch_size >= 64:
        return "High (16GB GPU)"
    elif batch_size >= 32:
        return "Good (8-12GB GPU)"
    elif batch_size >= 16:
        return "Moderate (4-8GB GPU or MPS)"
    elif batch_size >= 8:
        return "Low (Limited GPU)"
    else:
        return "Minimal (CPU or very limited GPU)"


# ============================================================================
# Memory Cleanup
# ============================================================================

def clear_gpu_cache():
    """
    Clear GPU memory cache to free up memory.

    Useful to call before large batch operations.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 CUDA cache cleared")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have explicit cache clearing
        logger.debug("MPS doesn't require manual cache clearing")
    else:
        logger.debug("No GPU cache to clear (CPU mode)")


def log_gpu_status():
    """Log current GPU status for debugging."""
    gpu_info = get_gpu_memory_info()

    if gpu_info.is_cuda:
        logger.info(
            f"📊 GPU Status: {gpu_info.device_name} | "
            f"Total: {gpu_info.total_memory_gb:.1f}GB | "
            f"Free: {gpu_info.free_memory_gb:.1f}GB | "
            f"Used: {gpu_info.utilization_percent:.1f}%"
        )
    elif gpu_info.is_mps:
        logger.info(f"📊 GPU Status: {gpu_info.device_name} (memory info not available)")
    else:
        logger.info(f"📊 Device Status: CPU mode")
