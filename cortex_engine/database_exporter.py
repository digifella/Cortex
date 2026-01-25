# ## File: cortex_engine/database_exporter.py
# Version: 1.0.0 (Database Portability)
# Date: 2026-01-25
# Purpose: Portable database export/import for cross-machine transfers.
#          - Creates zip packages with manifest for hardware compatibility checking
#          - Supports Qwen3-VL 2B/8B embedding model auto-configuration on import
#          - Validates hardware requirements before import

import json
import os
import shutil
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .utils import convert_windows_to_wsl_path, get_logger
from .utils.path_utils import _in_docker

logger = get_logger(__name__)


# ============================================================================
# EXPORT MANIFEST SCHEMA
# ============================================================================

@dataclass
class ExportManifest:
    """Manifest schema for portable database exports."""

    # Format versioning
    format_version: str = "1.0.0"

    # Cortex version that created this export
    cortex_version: str = ""

    # Export metadata
    export_name: str = ""
    export_date: str = ""
    source_machine: str = ""

    # Embedding configuration
    embedding_model: str = ""
    embedding_dimension: int = 0
    embedding_approach: str = ""  # "qwen3vl", "nv-embed", "bge"
    qwen3_vl_model_size: str = ""  # "2B", "8B", or ""

    # Database statistics
    document_count: int = 0
    collection_count: int = 0
    graph_node_count: int = 0
    graph_edge_count: int = 0

    # Hardware requirements
    minimum_vram_gb: float = 0.0

    # MRL compatibility (for future cross-model querying)
    supports_mrl_truncation: bool = False
    mrl_compatible_dimensions: List[int] = field(default_factory=list)
    compatible_models: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExportManifest":
        """Create manifest from dictionary."""
        # Handle missing fields gracefully
        return cls(
            format_version=data.get("format_version", "1.0.0"),
            cortex_version=data.get("cortex_version", ""),
            export_name=data.get("export_name", ""),
            export_date=data.get("export_date", ""),
            source_machine=data.get("source_machine", ""),
            embedding_model=data.get("embedding_model", ""),
            embedding_dimension=data.get("embedding_dimension", 0),
            embedding_approach=data.get("embedding_approach", ""),
            qwen3_vl_model_size=data.get("qwen3_vl_model_size", ""),
            document_count=data.get("document_count", 0),
            collection_count=data.get("collection_count", 0),
            graph_node_count=data.get("graph_node_count", 0),
            graph_edge_count=data.get("graph_edge_count", 0),
            minimum_vram_gb=data.get("minimum_vram_gb", 0.0),
            supports_mrl_truncation=data.get("supports_mrl_truncation", False),
            mrl_compatible_dimensions=data.get("mrl_compatible_dimensions", []),
            compatible_models=data.get("compatible_models", []),
        )


# ============================================================================
# DATABASE EXPORTER
# ============================================================================

class DatabaseExporter:
    """Creates portable zip packages from Cortex databases."""

    def __init__(self, db_path: str):
        """
        Initialize exporter with database path.

        Args:
            db_path: Path to the Cortex database (contains knowledge_hub_db/, etc.)
        """
        # Handle path conversion for WSL/Docker
        if _in_docker():
            self.db_path = Path(db_path)
        else:
            self.db_path = Path(convert_windows_to_wsl_path(db_path))

    def _get_embedding_info(self) -> Dict[str, Any]:
        """Get current embedding configuration."""
        try:
            from .config import get_embedding_strategy, QWEN3_VL_MODEL_SIZE

            strategy = get_embedding_strategy()
            approach = strategy.get("approach", "")

            # For Qwen3-VL, derive dimensions and VRAM from the actual model size config
            # (not from potentially stale strategy cache)
            if approach == "qwen3vl":
                size = QWEN3_VL_MODEL_SIZE
                size_config = {
                    "2B": {"dims": 2048, "vram": 5.0, "model": "Qwen/Qwen3-VL-Embedding-2B"},
                    "8B": {"dims": 4096, "vram": 16.0, "model": "Qwen/Qwen3-VL-Embedding-8B"},
                }
                if size in size_config:
                    cfg = size_config[size]
                    return {
                        "model": cfg["model"],
                        "dimensions": cfg["dims"],
                        "approach": "qwen3vl",
                        "qwen3_vl_size": size,
                        "vram_required": cfg["vram"],
                    }
                else:
                    # "auto" - check what's actually loaded or infer from GPU
                    dims = strategy.get("dimensions", 2048)
                    is_8b = dims == 4096
                    return {
                        "model": strategy.get("model", ""),
                        "dimensions": dims,
                        "approach": "qwen3vl",
                        "qwen3_vl_size": "8B" if is_8b else "2B",
                        "vram_required": 16.0 if is_8b else 5.0,
                    }
            else:
                return {
                    "model": strategy.get("model", ""),
                    "dimensions": strategy.get("dimensions", 0),
                    "approach": approach,
                    "qwen3_vl_size": "",
                    "vram_required": strategy.get("vram_required_gb", 0.0),
                }
        except Exception as e:
            logger.warning(f"Could not get embedding info: {e}")
            return {
                "model": "",
                "dimensions": 0,
                "approach": "unknown",
                "qwen3_vl_size": "",
                "vram_required": 0.0,
            }

    def _get_database_stats(self) -> Dict[str, int]:
        """Get database statistics (document count, collections, graph nodes)."""
        stats = {
            "document_count": 0,
            "collection_count": 0,
            "graph_node_count": 0,
            "graph_edge_count": 0,
        }

        # Count documents from ChromaDB
        chroma_path = self.db_path / "knowledge_hub_db"
        if chroma_path.exists():
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(chroma_path))
                collections = client.list_collections()
                stats["collection_count"] = len(collections)
                for coll in collections:
                    stats["document_count"] += coll.count()
            except Exception as e:
                logger.warning(f"Could not count ChromaDB documents: {e}")

        # Count graph nodes/edges
        graph_path = self.db_path / "knowledge_cortex.gpickle"
        if graph_path.exists():
            try:
                import pickle
                with open(graph_path, "rb") as f:
                    graph = pickle.load(f)
                stats["graph_node_count"] = graph.number_of_nodes()
                stats["graph_edge_count"] = graph.number_of_edges()
            except Exception as e:
                logger.warning(f"Could not count graph nodes: {e}")

        return stats

    def _get_mrl_compatibility(self, approach: str, dimension: int) -> Tuple[bool, List[int]]:
        """
        Determine MRL (Matryoshka Representation Learning) compatibility.

        Qwen3-VL models support MRL truncation to lower dimensions.
        """
        if approach != "qwen3vl":
            return False, []

        # Qwen3-VL MRL dimensions (powers of 2 up to max dimension)
        mrl_dims = [64, 128, 256, 512, 1024, 2048]
        if dimension == 4096:
            mrl_dims.append(4096)

        compatible_dims = [d for d in mrl_dims if d <= dimension]
        return True, compatible_dims

    def _get_compatible_models(self, approach: str, dimension: int) -> List[str]:
        """Get list of models compatible with this database."""
        compatible = []

        if approach == "qwen3vl":
            if dimension == 2048:
                compatible.append("Qwen/Qwen3-VL-Embedding-2B")
                compatible.append("Qwen/Qwen3-VL-Embedding-8B (via MRL truncation)")
            elif dimension == 4096:
                compatible.append("Qwen/Qwen3-VL-Embedding-8B")
        elif approach == "nv-embed":
            compatible.append("nvidia/NV-Embed-v2")
        elif approach == "bge":
            compatible.append("BAAI/bge-base-en-v1.5")

        return compatible

    def create_export(
        self,
        destination_folder: str,
        export_name: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[bool, str, Optional[ExportManifest]]:
        """
        Create a portable database export.

        Args:
            destination_folder: Folder to save the export zip
            export_name: Optional custom name for the export
            progress_callback: Optional callback(stage: str, percent: int)

        Returns:
            Tuple of (success, message, manifest)
        """
        try:
            # Validate source database
            if not self.db_path.exists():
                return False, f"Database path does not exist: {self.db_path}", None

            chroma_path = self.db_path / "knowledge_hub_db"
            if not chroma_path.exists():
                return False, "No ChromaDB database found in source path", None

            # Prepare destination
            if _in_docker():
                dest_path = Path(destination_folder)
            else:
                dest_path = Path(convert_windows_to_wsl_path(destination_folder))

            if not dest_path.exists():
                dest_path.mkdir(parents=True, exist_ok=True)

            # Generate export name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not export_name:
                export_name = f"cortex_export_{timestamp}"
            else:
                export_name = f"cortex_export_{export_name}_{timestamp}"

            zip_path = dest_path / f"{export_name}.zip"

            if progress_callback:
                progress_callback("Gathering metadata", 5)

            # Get embedding and database info
            embed_info = self._get_embedding_info()
            db_stats = self._get_database_stats()
            supports_mrl, mrl_dims = self._get_mrl_compatibility(
                embed_info["approach"], embed_info["dimensions"]
            )
            compatible_models = self._get_compatible_models(
                embed_info["approach"], embed_info["dimensions"]
            )

            # Get Cortex version
            try:
                from .version_config import CORTEX_VERSION
                cortex_version = CORTEX_VERSION
            except ImportError:
                cortex_version = "unknown"

            # Get machine name
            import socket
            source_machine = socket.gethostname()

            # Build manifest
            manifest = ExportManifest(
                format_version="1.0.0",
                cortex_version=cortex_version,
                export_name=export_name,
                export_date=datetime.now().isoformat(),
                source_machine=source_machine,
                embedding_model=embed_info["model"],
                embedding_dimension=embed_info["dimensions"],
                embedding_approach=embed_info["approach"],
                qwen3_vl_model_size=embed_info["qwen3_vl_size"],
                document_count=db_stats["document_count"],
                collection_count=db_stats["collection_count"],
                graph_node_count=db_stats["graph_node_count"],
                graph_edge_count=db_stats["graph_edge_count"],
                minimum_vram_gb=embed_info["vram_required"],
                supports_mrl_truncation=supports_mrl,
                mrl_compatible_dimensions=mrl_dims,
                compatible_models=compatible_models,
            )

            if progress_callback:
                progress_callback("Creating archive", 10)

            # Create zip archive
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add manifest
                manifest_json = json.dumps(manifest.to_dict(), indent=2)
                zf.writestr("export_manifest.json", manifest_json)

                if progress_callback:
                    progress_callback("Archiving ChromaDB", 20)

                # Add ChromaDB directory
                if chroma_path.exists():
                    for file_path in chroma_path.rglob("*"):
                        if file_path.is_file():
                            arcname = f"knowledge_hub_db/{file_path.relative_to(chroma_path)}"
                            zf.write(file_path, arcname)

                if progress_callback:
                    progress_callback("Archiving knowledge graph", 70)

                # Add knowledge graph
                graph_path = self.db_path / "knowledge_cortex.gpickle"
                if graph_path.exists():
                    zf.write(graph_path, "knowledge_cortex.gpickle")

                if progress_callback:
                    progress_callback("Archiving collections", 85)

                # Add working collections
                collections_path = self.db_path / "working_collections.json"
                if collections_path.exists():
                    zf.write(collections_path, "working_collections.json")

            if progress_callback:
                progress_callback("Complete", 100)

            # Get final size
            zip_size_mb = zip_path.stat().st_size / (1024 * 1024)

            success_msg = (
                f"Export created: {zip_path}\n"
                f"Size: {zip_size_mb:.1f} MB | "
                f"Documents: {db_stats['document_count']} | "
                f"Model: {embed_info['qwen3_vl_size'] or embed_info['approach']}"
            )

            logger.info(f"Database export created: {zip_path}")
            return True, success_msg, manifest

        except PermissionError as e:
            return False, f"Permission denied: {e}", None
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False, f"Export failed: {e}", None


# ============================================================================
# DATABASE IMPORTER
# ============================================================================

class DatabaseImporter:
    """Validates and imports portable database packages."""

    def __init__(self, zip_path: str):
        """
        Initialize importer with path to export zip.

        Args:
            zip_path: Path to the .zip export file
        """
        if _in_docker():
            self.zip_path = Path(zip_path)
        else:
            self.zip_path = Path(convert_windows_to_wsl_path(zip_path))

        self._manifest: Optional[ExportManifest] = None

    def validate(self) -> Tuple[bool, str, Optional[ExportManifest]]:
        """
        Validate the export package.

        Returns:
            Tuple of (is_valid, message, manifest)
        """
        # Check file exists
        if not self.zip_path.exists():
            return False, f"File not found: {self.zip_path}", None

        if not self.zip_path.suffix.lower() == ".zip":
            return False, "File must be a .zip archive", None

        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                # Check for manifest
                if "export_manifest.json" not in zf.namelist():
                    return False, "Not a valid Cortex export: missing export_manifest.json", None

                # Parse manifest
                manifest_data = json.loads(zf.read("export_manifest.json").decode("utf-8"))
                self._manifest = ExportManifest.from_dict(manifest_data)

                # Check for required files
                has_chroma = any(n.startswith("knowledge_hub_db/") for n in zf.namelist())
                if not has_chroma:
                    return False, "Export package missing ChromaDB data", None

                return True, "Valid Cortex export package", self._manifest

        except zipfile.BadZipFile:
            return False, "Invalid or corrupted zip file", None
        except json.JSONDecodeError:
            return False, "Invalid manifest JSON format", None
        except Exception as e:
            return False, f"Validation error: {e}", None

    def check_hardware_compatibility(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if local hardware can run this database.

        Returns:
            Tuple of (is_compatible, message, details)
        """
        if self._manifest is None:
            valid, msg, _ = self.validate()
            if not valid:
                return False, msg, {}

        manifest = self._manifest
        details = {
            "required_vram_gb": manifest.minimum_vram_gb,
            "available_vram_gb": 0.0,
            "required_model": manifest.embedding_model,
            "recommended_size": manifest.qwen3_vl_model_size,
            "gpu_name": "Unknown",
        }

        # Detect local GPU
        try:
            from .utils.smart_model_selector import detect_nvidia_gpu
            has_gpu, gpu_info = detect_nvidia_gpu()

            if has_gpu and gpu_info:
                details["available_vram_gb"] = gpu_info.get("memory_total_gb", 0.0)
                details["gpu_name"] = gpu_info.get("device_name", "Unknown")
            else:
                details["available_vram_gb"] = 0.0
                details["gpu_name"] = "No GPU detected"
        except Exception as e:
            logger.warning(f"Could not detect GPU: {e}")
            details["gpu_name"] = "Detection failed"

        # Check compatibility
        required = manifest.minimum_vram_gb
        available = details["available_vram_gb"]

        if available >= required:
            return True, f"Compatible: {available:.1f}GB VRAM available (need {required:.1f}GB)", details
        elif available > 0:
            # Check if we can use MRL truncation
            if manifest.supports_mrl_truncation and manifest.embedding_approach == "qwen3vl":
                return True, (
                    f"Compatible via smaller model: Your GPU has {available:.1f}GB "
                    f"(optimal: {required:.1f}GB). Will use 2B model."
                ), details
            return False, f"Insufficient VRAM: need {required:.1f}GB, have {available:.1f}GB", details
        else:
            return False, "No GPU detected. This database requires GPU for embedding queries.", details

    def import_database(
        self,
        destination_path: str,
        overwrite: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[bool, str]:
        """
        Import the database to the specified destination.

        Args:
            destination_path: Path where to import the database
            overwrite: Whether to overwrite existing database
            progress_callback: Optional callback(stage: str, percent: int)

        Returns:
            Tuple of (success, message)
        """
        # Validate first
        valid, msg, manifest = self.validate()
        if not valid:
            return False, msg

        try:
            if _in_docker():
                dest_path = Path(destination_path)
            else:
                dest_path = Path(convert_windows_to_wsl_path(destination_path))

            # Check for existing database
            chroma_dest = dest_path / "knowledge_hub_db"
            if chroma_dest.exists() and not overwrite:
                return False, "Database already exists at destination. Enable overwrite to replace."

            if progress_callback:
                progress_callback("Preparing destination", 5)

            # Create destination if needed
            dest_path.mkdir(parents=True, exist_ok=True)

            # Backup existing if overwriting
            if chroma_dest.exists() and overwrite:
                backup_name = f"knowledge_hub_db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_path = dest_path / backup_name
                shutil.move(str(chroma_dest), str(backup_path))
                logger.info(f"Existing database backed up to: {backup_path}")

            if progress_callback:
                progress_callback("Extracting archive", 15)

            # Extract archive
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                # Extract all except manifest (we'll handle that separately)
                for member in zf.namelist():
                    if member == "export_manifest.json":
                        continue

                    # Extract to destination
                    target_path = dest_path / member
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    if not member.endswith('/'):
                        with zf.open(member) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)

            if progress_callback:
                progress_callback("Configuring embedding model", 85)

            # Auto-configure embedding model based on manifest
            if manifest.embedding_approach == "qwen3vl" and manifest.qwen3_vl_model_size:
                os.environ["QWEN3_VL_MODEL_SIZE"] = manifest.qwen3_vl_model_size
                os.environ["QWEN3_VL_ENABLED"] = "true"
                logger.info(f"Configured Qwen3-VL model size: {manifest.qwen3_vl_model_size}")

                # Invalidate caches
                try:
                    from .config import invalidate_embedding_cache
                    invalidate_embedding_cache()
                except Exception:
                    pass

                # Reset Qwen3-VL service
                try:
                    from .qwen3_vl_embedding_service import reset_service
                    reset_service()
                except Exception:
                    pass

            if progress_callback:
                progress_callback("Complete", 100)

            success_msg = (
                f"Database imported successfully to: {dest_path}\n"
                f"Documents: {manifest.document_count} | "
                f"Model: {manifest.qwen3_vl_model_size or manifest.embedding_approach} | "
                f"Dimensions: {manifest.embedding_dimension}"
            )

            logger.info(f"Database imported from {self.zip_path} to {dest_path}")
            return True, success_msg

        except PermissionError as e:
            return False, f"Permission denied: {e}"
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False, f"Import failed: {e}"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_export_summary(manifest: ExportManifest) -> Dict[str, Any]:
    """Get a human-readable summary of an export manifest."""
    return {
        "name": manifest.export_name,
        "date": manifest.export_date,
        "source": manifest.source_machine,
        "documents": manifest.document_count,
        "collections": manifest.collection_count,
        "embedding_model": manifest.embedding_model,
        "model_size": manifest.qwen3_vl_model_size or "N/A",
        "dimensions": manifest.embedding_dimension,
        "vram_required": f"{manifest.minimum_vram_gb:.1f} GB",
        "cortex_version": manifest.cortex_version,
        "mrl_compatible": manifest.supports_mrl_truncation,
    }
