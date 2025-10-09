"""
Performance Monitoring System
Tracks and reports performance metrics for critical operations including:
- Image processing (VLM)
- Embedding generation (batch processing)
- Query caching (hit/miss rates)
- Overall ingestion throughput

Metrics are persisted to JSON and can be viewed in the Maintenance dashboard.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from contextlib import contextmanager

from .logging_utils import get_logger
from .default_paths import get_default_ai_database_path

logger = get_logger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class OperationMetric:
    """Single operation timing metric"""
    operation_type: str  # "image_processing", "embedding_batch", "query", etc.
    start_time: float
    end_time: float
    duration: float  # seconds
    success: bool
    metadata: Dict[str, Any]  # Additional context (batch_size, cache_hit, etc.)
    timestamp: str  # ISO format


@dataclass
class PerformanceStats:
    """Aggregated performance statistics"""
    operation_type: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    total_duration: float  # seconds
    avg_duration: float  # seconds
    min_duration: float  # seconds
    max_duration: float  # seconds
    p50_duration: float  # median
    p95_duration: float  # 95th percentile
    p99_duration: float  # 99th percentile
    first_seen: str  # ISO timestamp
    last_seen: str  # ISO timestamp


# ============================================================================
# Performance Monitor (Singleton)
# ============================================================================

class PerformanceMonitor:
    """
    Thread-safe performance monitoring system.

    Tracks timing metrics for critical operations and provides
    aggregated statistics for performance analysis.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))  # Keep last 1000 per type
        self._metrics_lock = threading.Lock()

        # Session tracking
        self._session_start = time.time()
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"📊 Performance monitor initialized (session: {self._session_id})")

    def record_operation(
        self,
        operation_type: str,
        duration: float,
        success: bool = True,
        **metadata
    ):
        """
        Record a completed operation.

        Args:
            operation_type: Type of operation (e.g., "image_processing", "embedding_batch")
            duration: Duration in seconds
            success: Whether operation succeeded
            **metadata: Additional context (batch_size, item_count, cache_hit, etc.)
        """
        metric = OperationMetric(
            operation_type=operation_type,
            start_time=time.time() - duration,
            end_time=time.time(),
            duration=duration,
            success=success,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )

        with self._metrics_lock:
            self._metrics[operation_type].append(metric)

        # Log significant operations
        if duration > 5.0:  # Operations > 5 seconds
            logger.info(f"⏱️ {operation_type}: {duration:.2f}s (metadata: {metadata})")

    @contextmanager
    def measure(self, operation_type: str, **metadata):
        """
        Context manager for timing operations.

        Usage:
            with monitor.measure("embedding_batch", batch_size=32, doc_count=100):
                # ... perform operation ...
                pass
        """
        start = time.time()
        success = True
        try:
            yield
        except Exception as e:
            success = False
            raise
        finally:
            duration = time.time() - start
            self.record_operation(operation_type, duration, success, **metadata)

    def get_stats(self, operation_type: str) -> Optional[PerformanceStats]:
        """
        Get aggregated statistics for an operation type.

        Args:
            operation_type: Type of operation

        Returns:
            PerformanceStats or None if no data
        """
        with self._metrics_lock:
            metrics = list(self._metrics.get(operation_type, []))

        if not metrics:
            return None

        # Calculate statistics
        durations = [m.duration for m in metrics]
        durations_sorted = sorted(durations)
        successful = [m for m in metrics if m.success]
        failed = [m for m in metrics if not m.success]

        n = len(durations_sorted)

        return PerformanceStats(
            operation_type=operation_type,
            total_operations=len(metrics),
            successful_operations=len(successful),
            failed_operations=len(failed),
            total_duration=sum(durations),
            avg_duration=sum(durations) / len(durations),
            min_duration=min(durations),
            max_duration=max(durations),
            p50_duration=durations_sorted[n // 2] if n > 0 else 0.0,
            p95_duration=durations_sorted[int(n * 0.95)] if n > 1 else durations_sorted[0],
            p99_duration=durations_sorted[int(n * 0.99)] if n > 1 else durations_sorted[0],
            first_seen=metrics[0].timestamp,
            last_seen=metrics[-1].timestamp
        )

    def get_all_stats(self) -> Dict[str, PerformanceStats]:
        """Get statistics for all tracked operation types."""
        with self._metrics_lock:
            operation_types = list(self._metrics.keys())

        return {
            op_type: self.get_stats(op_type)
            for op_type in operation_types
            if self.get_stats(op_type) is not None
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get query cache hit/miss statistics."""
        with self._metrics_lock:
            query_metrics = list(self._metrics.get("query", []))

        if not query_metrics:
            return {
                "total_queries": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "hit_rate": 0.0
            }

        cache_hits = sum(1 for m in query_metrics if m.metadata.get("cache_hit", False))
        cache_misses = len(query_metrics) - cache_hits

        return {
            "total_queries": len(query_metrics),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "hit_rate": cache_hits / len(query_metrics) if query_metrics else 0.0
        }

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current monitoring session."""
        session_duration = time.time() - self._session_start
        all_stats = self.get_all_stats()
        cache_stats = self.get_cache_stats()

        return {
            "session_id": self._session_id,
            "session_duration_seconds": session_duration,
            "session_duration_formatted": f"{session_duration / 60:.1f} minutes",
            "operation_types_tracked": len(all_stats),
            "total_operations": sum(s.total_operations for s in all_stats.values()),
            "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
            "statistics": {k: asdict(v) for k, v in all_stats.items()},
            "cache_statistics": cache_stats
        }

    def save_to_file(self, file_path: Optional[Path] = None) -> Path:
        """
        Save performance metrics to JSON file.

        Args:
            file_path: Optional custom path, defaults to DB path

        Returns:
            Path to saved file
        """
        if file_path is None:
            db_path = Path(get_default_ai_database_path())
            db_path.mkdir(parents=True, exist_ok=True)
            file_path = db_path / f"performance_metrics_{self._session_id}.json"

        summary = self.get_session_summary()

        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"💾 Performance metrics saved to: {file_path}")
        return file_path

    def clear(self):
        """Clear all metrics (start fresh session)."""
        with self._metrics_lock:
            self._metrics.clear()
            self._session_start = time.time()
            self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"🔄 Performance metrics cleared (new session: {self._session_id})")

    def get_recent_metrics(
        self,
        operation_type: str,
        limit: int = 10
    ) -> List[OperationMetric]:
        """Get most recent metrics for an operation type."""
        with self._metrics_lock:
            metrics = list(self._metrics.get(operation_type, []))

        return metrics[-limit:] if metrics else []


# ============================================================================
# Global Instance
# ============================================================================

# Singleton instance for easy access
_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _monitor


# ============================================================================
# Convenience Functions
# ============================================================================

def record_operation(operation_type: str, duration: float, success: bool = True, **metadata):
    """Convenience function to record an operation."""
    _monitor.record_operation(operation_type, duration, success, **metadata)


@contextmanager
def measure(operation_type: str, **metadata):
    """Convenience context manager for timing operations."""
    with _monitor.measure(operation_type, **metadata):
        yield


def get_stats(operation_type: str) -> Optional[PerformanceStats]:
    """Convenience function to get stats."""
    return _monitor.get_stats(operation_type)


def get_all_stats() -> Dict[str, PerformanceStats]:
    """Convenience function to get all stats."""
    return _monitor.get_all_stats()


def get_session_summary() -> Dict[str, Any]:
    """Convenience function to get session summary."""
    return _monitor.get_session_summary()


def save_metrics(file_path: Optional[Path] = None) -> Path:
    """Convenience function to save metrics."""
    return _monitor.save_to_file(file_path)
