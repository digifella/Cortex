"""
Resource Throttler for Long-Running Operations
===============================================

Version: 1.0.0
Date: 2026-01-26

Provides system resource monitoring and throttling to prevent runaway
memory/GPU usage during embedding, indexing, and other intensive operations.

CRITICAL: This module exists because uncontrolled batch processing can
drive GPU to 100% AND CPU memory to 99%, nearly crashing the system.

Key Features:
- CPU (system) memory monitoring via psutil
- GPU memory monitoring via PyTorch
- Automatic throttling when resources get tight
- Pause/resume capability for long-running batches
- Configurable thresholds for different safety levels
"""

from __future__ import annotations

import os
import time
import gc
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum

import torch

from .logging_utils import get_logger

logger = get_logger(__name__)

# Try to import psutil for CPU memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed - CPU memory monitoring disabled. Install with: pip install psutil")


class ThrottleLevel(Enum):
    """Throttle intensity levels."""
    NONE = "none"           # No throttling needed
    LIGHT = "light"         # Minor delay between batches
    MODERATE = "moderate"   # Longer delay + cache clearing
    HEAVY = "heavy"         # Reduce batch size + significant delay
    CRITICAL = "critical"   # Pause processing, wait for resources


@dataclass
class ResourceState:
    """Current system resource state."""
    cpu_memory_percent: float
    cpu_memory_available_gb: float
    gpu_memory_percent: float
    gpu_memory_free_gb: float
    gpu_memory_total_gb: float
    throttle_level: ThrottleLevel
    is_safe: bool
    warnings: list


@dataclass
class ThrottleConfig:
    """Configuration for resource throttling."""
    # CPU memory thresholds (percent used)
    cpu_warning_threshold: float = 75.0
    cpu_danger_threshold: float = 85.0
    cpu_critical_threshold: float = 92.0

    # GPU memory thresholds (percent used)
    gpu_warning_threshold: float = 80.0
    gpu_danger_threshold: float = 90.0
    gpu_critical_threshold: float = 95.0

    # Throttle delays (seconds)
    light_delay: float = 0.5
    moderate_delay: float = 2.0
    heavy_delay: float = 5.0
    critical_delay: float = 10.0

    # Batch size reduction factors
    moderate_batch_factor: float = 0.75
    heavy_batch_factor: float = 0.5
    critical_batch_factor: float = 0.25

    # How often to check resources (every N batches)
    check_interval: int = 1  # Check every batch for safety

    # Enable/disable specific features
    enable_cpu_monitoring: bool = True
    enable_gpu_monitoring: bool = True
    enable_cache_clearing: bool = True
    enable_gc_collection: bool = True

    @classmethod
    def conservative(cls) -> "ThrottleConfig":
        """Very conservative settings for maximum stability."""
        return cls(
            cpu_warning_threshold=65.0,
            cpu_danger_threshold=75.0,
            cpu_critical_threshold=85.0,
            gpu_warning_threshold=70.0,
            gpu_danger_threshold=80.0,
            gpu_critical_threshold=90.0,
            light_delay=1.0,
            moderate_delay=3.0,
            heavy_delay=8.0,
            critical_delay=15.0,
        )

    @classmethod
    def balanced(cls) -> "ThrottleConfig":
        """Balanced settings for typical use."""
        return cls()  # Default values

    @classmethod
    def performance(cls) -> "ThrottleConfig":
        """Performance-oriented settings (less throttling)."""
        return cls(
            cpu_warning_threshold=80.0,
            cpu_danger_threshold=90.0,
            cpu_critical_threshold=95.0,
            gpu_warning_threshold=85.0,
            gpu_danger_threshold=92.0,
            gpu_critical_threshold=97.0,
            light_delay=0.2,
            moderate_delay=1.0,
            heavy_delay=3.0,
            critical_delay=5.0,
        )


class ResourceThrottler:
    """
    Monitors system resources and applies throttling during long-running operations.

    Usage:
        throttler = ResourceThrottler(ThrottleConfig.conservative())

        for batch in batches:
            # Check resources before processing
            state = throttler.check_and_throttle()

            if state.throttle_level == ThrottleLevel.CRITICAL:
                logger.warning("Resources critical - pausing...")
                time.sleep(throttler.config.critical_delay)
                continue

            # Get adjusted batch size
            adjusted_size = throttler.get_adjusted_batch_size(original_size)

            # Process batch...
            process_batch(batch[:adjusted_size])

            # Apply post-batch throttling
            throttler.post_batch_cleanup()
    """

    def __init__(self, config: Optional[ThrottleConfig] = None):
        """
        Initialize throttler with configuration.

        Args:
            config: Throttle configuration. Uses conservative defaults if None.
        """
        self.config = config or ThrottleConfig.conservative()
        self._batch_count = 0
        self._total_paused_time = 0.0
        self._warnings_issued = 0

    def get_resource_state(self) -> ResourceState:
        """
        Get current system resource state.

        Returns:
            ResourceState with current memory usage and throttle recommendation
        """
        warnings = []

        # CPU memory monitoring
        cpu_percent = 0.0
        cpu_available_gb = 0.0
        if PSUTIL_AVAILABLE and self.config.enable_cpu_monitoring:
            mem = psutil.virtual_memory()
            cpu_percent = mem.percent
            cpu_available_gb = mem.available / (1024 ** 3)

        # GPU memory monitoring
        gpu_percent = 0.0
        gpu_free_gb = 0.0
        gpu_total_gb = 0.0
        if torch.cuda.is_available() and self.config.enable_gpu_monitoring:
            gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            gpu_free_gb = gpu_total_gb - gpu_reserved
            gpu_percent = (gpu_reserved / gpu_total_gb * 100) if gpu_total_gb > 0 else 0.0

        # Determine throttle level based on highest concern
        throttle_level = ThrottleLevel.NONE
        is_safe = True

        # Check CPU thresholds
        if cpu_percent >= self.config.cpu_critical_threshold:
            throttle_level = ThrottleLevel.CRITICAL
            is_safe = False
            warnings.append(f"CRITICAL: CPU memory at {cpu_percent:.1f}% (threshold: {self.config.cpu_critical_threshold}%)")
        elif cpu_percent >= self.config.cpu_danger_threshold:
            throttle_level = max(throttle_level, ThrottleLevel.HEAVY, key=lambda x: list(ThrottleLevel).index(x))
            is_safe = False
            warnings.append(f"DANGER: CPU memory at {cpu_percent:.1f}%")
        elif cpu_percent >= self.config.cpu_warning_threshold:
            throttle_level = max(throttle_level, ThrottleLevel.MODERATE, key=lambda x: list(ThrottleLevel).index(x))
            warnings.append(f"WARNING: CPU memory at {cpu_percent:.1f}%")

        # Check GPU thresholds
        if gpu_percent >= self.config.gpu_critical_threshold:
            throttle_level = ThrottleLevel.CRITICAL
            is_safe = False
            warnings.append(f"CRITICAL: GPU memory at {gpu_percent:.1f}% (threshold: {self.config.gpu_critical_threshold}%)")
        elif gpu_percent >= self.config.gpu_danger_threshold:
            if throttle_level != ThrottleLevel.CRITICAL:
                throttle_level = ThrottleLevel.HEAVY
            is_safe = False
            warnings.append(f"DANGER: GPU memory at {gpu_percent:.1f}%")
        elif gpu_percent >= self.config.gpu_warning_threshold:
            if throttle_level not in [ThrottleLevel.CRITICAL, ThrottleLevel.HEAVY]:
                throttle_level = ThrottleLevel.MODERATE
            warnings.append(f"WARNING: GPU memory at {gpu_percent:.1f}%")

        return ResourceState(
            cpu_memory_percent=cpu_percent,
            cpu_memory_available_gb=cpu_available_gb,
            gpu_memory_percent=gpu_percent,
            gpu_memory_free_gb=gpu_free_gb,
            gpu_memory_total_gb=gpu_total_gb,
            throttle_level=throttle_level,
            is_safe=is_safe,
            warnings=warnings
        )

    def check_and_throttle(self, force_check: bool = False) -> ResourceState:
        """
        Check resources and apply throttling if needed.

        Args:
            force_check: Force check even if not at check interval

        Returns:
            Current ResourceState
        """
        self._batch_count += 1

        # Only check at intervals unless forced
        if not force_check and (self._batch_count % self.config.check_interval != 0):
            return ResourceState(
                cpu_memory_percent=0,
                cpu_memory_available_gb=0,
                gpu_memory_percent=0,
                gpu_memory_free_gb=0,
                gpu_memory_total_gb=0,
                throttle_level=ThrottleLevel.NONE,
                is_safe=True,
                warnings=[]
            )

        state = self.get_resource_state()

        # Log warnings
        for warning in state.warnings:
            logger.warning(f"🚨 {warning}")
            self._warnings_issued += 1

        # Apply delays based on throttle level
        delay = 0.0
        if state.throttle_level == ThrottleLevel.LIGHT:
            delay = self.config.light_delay
        elif state.throttle_level == ThrottleLevel.MODERATE:
            delay = self.config.moderate_delay
        elif state.throttle_level == ThrottleLevel.HEAVY:
            delay = self.config.heavy_delay
        elif state.throttle_level == ThrottleLevel.CRITICAL:
            delay = self.config.critical_delay
            logger.error(f"🛑 CRITICAL resource state - pausing for {delay}s")

        if delay > 0:
            logger.info(f"⏸️ Throttling: {state.throttle_level.value} - pausing {delay:.1f}s")
            time.sleep(delay)
            self._total_paused_time += delay

            # After pause, do cleanup
            self._cleanup_memory()

        return state

    def get_adjusted_batch_size(self, original_size: int, state: Optional[ResourceState] = None) -> int:
        """
        Get adjusted batch size based on current resource state.

        Args:
            original_size: Original requested batch size
            state: Resource state (will fetch if not provided)

        Returns:
            Adjusted batch size (may be smaller than original)
        """
        if state is None:
            state = self.get_resource_state()

        factor = 1.0
        if state.throttle_level == ThrottleLevel.MODERATE:
            factor = self.config.moderate_batch_factor
        elif state.throttle_level == ThrottleLevel.HEAVY:
            factor = self.config.heavy_batch_factor
        elif state.throttle_level == ThrottleLevel.CRITICAL:
            factor = self.config.critical_batch_factor

        adjusted = max(1, int(original_size * factor))

        if adjusted < original_size:
            logger.info(f"📉 Batch size reduced: {original_size} → {adjusted} (factor: {factor})")

        return adjusted

    def post_batch_cleanup(self):
        """
        Perform cleanup after processing a batch.

        Call this after each batch to manage memory proactively.
        """
        self._cleanup_memory()

    def _cleanup_memory(self):
        """Internal memory cleanup."""
        if self.config.enable_gc_collection:
            gc.collect()

        if self.config.enable_cache_clearing and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get throttler statistics.

        Returns:
            Dictionary with throttling stats
        """
        return {
            "batches_processed": self._batch_count,
            "total_paused_time_seconds": self._total_paused_time,
            "warnings_issued": self._warnings_issued,
            "config": {
                "cpu_critical_threshold": self.config.cpu_critical_threshold,
                "gpu_critical_threshold": self.config.gpu_critical_threshold,
            }
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self._batch_count = 0
        self._total_paused_time = 0.0
        self._warnings_issued = 0


# ============================================================================
# Convenience Functions
# ============================================================================

_default_throttler: Optional[ResourceThrottler] = None


def get_default_throttler() -> ResourceThrottler:
    """Get or create the default throttler instance."""
    global _default_throttler
    if _default_throttler is None:
        _default_throttler = ResourceThrottler(ThrottleConfig.conservative())
    return _default_throttler


def check_resources() -> ResourceState:
    """Quick check of current resource state using default throttler."""
    return get_default_throttler().get_resource_state()


def throttle_if_needed() -> ResourceState:
    """Check and apply throttling if needed using default throttler."""
    return get_default_throttler().check_and_throttle(force_check=True)


def log_resource_status():
    """Log current resource status for debugging."""
    state = check_resources()

    logger.info(f"📊 Resource Status:")
    if PSUTIL_AVAILABLE:
        logger.info(f"   CPU Memory: {state.cpu_memory_percent:.1f}% used, {state.cpu_memory_available_gb:.1f}GB available")
    if state.gpu_memory_total_gb > 0:
        logger.info(f"   GPU Memory: {state.gpu_memory_percent:.1f}% used, {state.gpu_memory_free_gb:.1f}GB free of {state.gpu_memory_total_gb:.1f}GB")
    logger.info(f"   Throttle Level: {state.throttle_level.value}")
    logger.info(f"   Safe to proceed: {state.is_safe}")


def wait_for_safe_resources(timeout_seconds: float = 120.0, check_interval: float = 5.0) -> bool:
    """
    Wait until resources are at safe levels.

    Args:
        timeout_seconds: Maximum time to wait
        check_interval: How often to check

    Returns:
        True if resources became safe, False if timeout
    """
    start_time = time.time()
    throttler = get_default_throttler()

    while (time.time() - start_time) < timeout_seconds:
        state = throttler.get_resource_state()

        if state.is_safe:
            logger.info("✅ Resources at safe levels - proceeding")
            return True

        elapsed = time.time() - start_time
        logger.warning(f"⏳ Waiting for resources... ({elapsed:.0f}s / {timeout_seconds:.0f}s)")

        # Cleanup while waiting
        throttler._cleanup_memory()
        time.sleep(check_interval)

    logger.error(f"⏰ Timeout waiting for safe resources after {timeout_seconds}s")
    return False
