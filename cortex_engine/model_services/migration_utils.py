"""
Model Migration Utilities
Tools for migrating models between different backends (Ollama <-> Docker Model Runner).
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
import json
import tempfile
import shutil

from .interfaces import ModelInfo, ModelStatus, ModelDownloadProgress
from .hybrid_manager import HybridModelManager
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class MigrationPlan:
    """Plan for migrating models between backends."""
    source_backend: str
    target_backend: str
    models_to_migrate: List[ModelInfo]
    estimated_time_minutes: int
    estimated_size_gb: float
    warnings: List[str]
    
    @property
    def summary(self) -> str:
        return (f"Migrate {len(self.models_to_migrate)} models from "
                f"{self.source_backend} to {self.target_backend} "
                f"(~{self.estimated_size_gb:.1f}GB, ~{self.estimated_time_minutes}min)")


@dataclass
class MigrationProgress:
    """Progress information for model migration."""
    plan: MigrationPlan
    current_model: Optional[str] = None
    completed_models: List[str] = None
    failed_models: List[str] = None
    status: str = "preparing"
    progress_percent: float = 0.0
    
    def __post_init__(self):
        if self.completed_models is None:
            self.completed_models = []
        if self.failed_models is None:
            self.failed_models = []


class ModelMigrationManager:
    """Manages model migration between different backends."""
    
    def __init__(self, hybrid_manager: HybridModelManager):
        self.hybrid_manager = hybrid_manager
        
        # Migration mappings
        self._model_name_mappings = {
            # Ollama -> Docker Model Runner
            "mistral:7b-instruct-v0.3-q4_K_M": "ai/mistral:7b-instruct-v0.3-q4_K_M",
            "mistral": "ai/mistral:7b-instruct",
            "mistral-small3.2": "ai/mistral:small-3.2",
            "llava": "ai/llava:latest",
            "codellama": "ai/codellama:latest",
            "phi": "ai/phi:latest"
        }
        
        # Reverse mappings (Docker -> Ollama)
        self._reverse_mappings = {v: k for k, v in self._model_name_mappings.items()}
    
    def _get_target_model_name(self, source_name: str, source_backend: str, target_backend: str) -> str:
        """Get the equivalent model name in the target backend."""
        if source_backend == "ollama" and target_backend == "docker_model_runner":
            return self._model_name_mappings.get(source_name, f"ai/{source_name}")
        elif source_backend == "docker_model_runner" and target_backend == "ollama":
            return self._reverse_mappings.get(source_name, source_name.replace("ai/", ""))
        else:
            return source_name
    
    async def create_migration_plan(self, 
                                  source_backend: str, 
                                  target_backend: str,
                                  models: Optional[List[str]] = None) -> MigrationPlan:
        """Create a migration plan for moving models between backends."""
        
        source_service = await self.hybrid_manager._get_backend_service(source_backend)
        target_service = await self.hybrid_manager._get_backend_service(target_backend)
        
        if not source_service:
            raise ValueError(f"Source backend {source_backend} not available")
        if not target_service:
            raise ValueError(f"Target backend {target_backend} not available")
        
        # Get available models from source
        available_models = await source_service.list_available_models()
        
        # Filter models if specified
        if models:
            available_models = [m for m in available_models if m.full_name in models or m.name in models]
        
        # Check which models need migration
        models_to_migrate = []
        warnings = []
        total_size_gb = 0.0
        
        for model in available_models:
            target_name = self._get_target_model_name(model.full_name, source_backend, target_backend)
            
            # Check if model already exists in target
            if await target_service.is_model_available(target_name):
                warnings.append(f"Model {target_name} already exists in {target_backend}")
                continue
            
            # Check if mapping exists
            if source_backend == "ollama" and target_backend == "docker_model_runner":
                if model.full_name not in self._model_name_mappings:
                    warnings.append(f"No Docker equivalent known for {model.full_name}")
                    continue
            
            models_to_migrate.append(model)
            total_size_gb += model.size_gb
        
        # Estimate time (rough: 1GB per minute for download + processing overhead)
        estimated_time = max(10, int(total_size_gb * 1.5))
        
        return MigrationPlan(
            source_backend=source_backend,
            target_backend=target_backend,
            models_to_migrate=models_to_migrate,
            estimated_time_minutes=estimated_time,
            estimated_size_gb=total_size_gb,
            warnings=warnings
        )
    
    async def execute_migration(self, 
                              migration_plan: MigrationPlan,
                              keep_source: bool = True,
                              verify_after: bool = True) -> AsyncIterator[MigrationProgress]:
        """Execute a migration plan."""
        
        progress = MigrationProgress(plan=migration_plan)
        yield progress
        
        target_service = await self.hybrid_manager._get_backend_service(migration_plan.target_backend)
        source_service = await self.hybrid_manager._get_backend_service(migration_plan.source_backend)
        
        if not target_service or not source_service:
            progress.status = "Error: Backend services not available"
            yield progress
            return
        
        total_models = len(migration_plan.models_to_migrate)
        
        for i, model in enumerate(migration_plan.models_to_migrate):
            progress.current_model = model.full_name
            progress.status = f"Migrating {model.full_name}..."
            progress.progress_percent = (i / total_models) * 90  # Reserve 10% for verification
            yield progress
            
            try:
                target_name = self._get_target_model_name(
                    model.full_name, 
                    migration_plan.source_backend, 
                    migration_plan.target_backend
                )
                
                # Pull model in target backend
                migration_success = False
                async for download_progress in target_service.pull_model(target_name):
                    progress.status = f"Migrating {model.full_name}: {download_progress.status}"
                    yield progress
                    
                    # Check if download completed successfully
                    if "success" in download_progress.status.lower() or "completed" in download_progress.status.lower():
                        migration_success = True
                        break
                
                if migration_success:
                    progress.completed_models.append(model.full_name)
                    
                    # Verify the model works
                    if verify_after:
                        progress.status = f"Verifying {model.full_name}..."
                        yield progress
                        
                        if not await target_service.test_model_inference(target_name):
                            progress.failed_models.append(f"{model.full_name} (verification failed)")
                            logger.warning(f"Model {target_name} migration completed but verification failed")
                        else:
                            logger.info(f"Model {target_name} migrated and verified successfully")
                    
                    # Remove from source if requested
                    if not keep_source:
                        progress.status = f"Removing {model.full_name} from source..."
                        yield progress
                        
                        if await source_service.remove_model(model.full_name):
                            logger.info(f"Removed {model.full_name} from {migration_plan.source_backend}")
                        else:
                            logger.warning(f"Failed to remove {model.full_name} from source")
                
                else:
                    progress.failed_models.append(model.full_name)
                    logger.error(f"Failed to migrate {model.full_name}")
                    
            except Exception as e:
                progress.failed_models.append(f"{model.full_name} (error: {str(e)})")
                logger.error(f"Migration failed for {model.full_name}: {e}")
        
        # Final status
        progress.current_model = None
        progress.progress_percent = 100.0
        
        if progress.failed_models:
            progress.status = f"Migration completed with {len(progress.failed_models)} failures"
        else:
            progress.status = "Migration completed successfully"
        
        yield progress
    
    async def get_migration_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for model migration based on current setup."""
        
        system_status = await self.hybrid_manager.get_system_status()
        recommendations = {
            "current_strategy": self.hybrid_manager.strategy.value,
            "available_backends": system_status.get("available_backends", []),
            "recommendations": []
        }
        
        # Get models in each backend
        backend_models = {}
        for backend_name in system_status.get("available_backends", []):
            service = await self.hybrid_manager._get_backend_service(backend_name)
            if service:
                models = await service.list_available_models()
                backend_models[backend_name] = [m.full_name for m in models]
        
        # Recommendation logic
        if "docker_model_runner" in backend_models and "ollama" in backend_models:
            docker_models = set(backend_models["docker_model_runner"])
            ollama_models = set(backend_models["ollama"])
            
            # Find models only in Ollama that could benefit from Docker Model Runner
            ollama_only = ollama_models - docker_models
            docker_equivalent_models = [m for m in ollama_only if m in self._model_name_mappings]
            
            if docker_equivalent_models:
                recommendations["recommendations"].append({
                    "type": "performance_upgrade",
                    "description": "Migrate enterprise models to Docker Model Runner for better performance",
                    "models": docker_equivalent_models,
                    "benefit": "15% faster inference, better GPU utilization",
                    "action": "migrate_to_docker"
                })
            
            # Find models only in Docker that might need Ollama compatibility
            docker_only = docker_models - ollama_models
            if docker_only and len(ollama_models) > 0:
                recommendations["recommendations"].append({
                    "type": "compatibility",
                    "description": "Some models only available in Docker Model Runner",
                    "models": list(docker_only),
                    "benefit": "Maintains compatibility with existing workflows",
                    "action": "consider_ollama_alternatives"
                })
        
        elif "ollama" in backend_models and "docker_model_runner" not in backend_models:
            recommendations["recommendations"].append({
                "type": "enterprise_upgrade",
                "description": "Enable Docker Model Runner for enterprise features",
                "benefit": "Better performance, enterprise compliance, OCI distribution",
                "action": "setup_docker_model_runner"
            })
        
        elif "docker_model_runner" in backend_models and "ollama" not in backend_models:
            recommendations["recommendations"].append({
                "type": "compatibility_fallback",
                "description": "Enable Ollama for broader model compatibility",
                "benefit": "Access to community models, development flexibility",
                "action": "setup_ollama_fallback"
            })
        
        return recommendations
    
    async def export_migration_report(self, 
                                    migration_progress: MigrationProgress,
                                    output_path: Path) -> bool:
        """Export a detailed migration report."""
        try:
            report = {
                "migration_summary": {
                    "source_backend": migration_progress.plan.source_backend,
                    "target_backend": migration_progress.plan.target_backend,
                    "total_models": len(migration_progress.plan.models_to_migrate),
                    "successful_migrations": len(migration_progress.completed_models),
                    "failed_migrations": len(migration_progress.failed_models),
                    "estimated_size_gb": migration_progress.plan.estimated_size_gb,
                    "estimated_time_minutes": migration_progress.plan.estimated_time_minutes
                },
                "completed_models": migration_progress.completed_models,
                "failed_models": migration_progress.failed_models,
                "warnings": migration_progress.plan.warnings,
                "final_status": migration_progress.status,
                "recommendations": await self.get_migration_recommendations()
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Migration report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export migration report: {e}")
            return False
    
    async def validate_migration_compatibility(self, 
                                             source_backend: str, 
                                             target_backend: str) -> Dict[str, Any]:
        """Validate that migration between backends is possible."""
        
        validation = {
            "compatible": False,
            "source_available": False,
            "target_available": False,
            "supported_direction": False,
            "issues": []
        }
        
        # Check if backends are available
        availability = await self.hybrid_manager._check_backend_availability()
        validation["source_available"] = availability.get(source_backend, False)
        validation["target_available"] = availability.get(target_backend, False)
        
        if not validation["source_available"]:
            validation["issues"].append(f"Source backend {source_backend} is not available")
        
        if not validation["target_available"]:
            validation["issues"].append(f"Target backend {target_backend} is not available")
        
        # Check if migration direction is supported
        supported_migrations = [
            ("ollama", "docker_model_runner"),
            ("docker_model_runner", "ollama")
        ]
        
        if (source_backend, target_backend) in supported_migrations:
            validation["supported_direction"] = True
        else:
            validation["issues"].append(f"Migration from {source_backend} to {target_backend} is not supported")
        
        # Overall compatibility
        validation["compatible"] = (validation["source_available"] and 
                                  validation["target_available"] and 
                                  validation["supported_direction"])
        
        return validation