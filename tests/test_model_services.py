"""
Test Suite for Hybrid Model Services
Tests the Docker Model Runner and Ollama integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from cortex_engine.model_services import (
    HybridModelManager,
    DockerModelService, 
    OllamaModelService,
    ModelRegistry,
    DistributionStrategy
)
from cortex_engine.model_services.interfaces import ModelInfo, ModelStatus
from cortex_engine.model_services.migration_utils import ModelMigrationManager


class TestModelRegistry:
    """Test the model registry functionality."""
    
    def test_registry_initialization(self):
        """Test that registry initializes with default models."""
        registry = ModelRegistry()
        
        # Should have default models
        mistral_entry = registry.get_model_entry("mistral")
        assert mistral_entry is not None
        assert mistral_entry.name == "mistral-7b-instruct"
        
        # Test aliases
        mistral_alias = registry.get_model_entry("mistral:7b-instruct-v0.3-q4_K_M")
        assert mistral_alias == mistral_entry
    
    def test_model_recommendations(self):
        """Test model recommendation system."""
        registry = ModelRegistry()
        
        # Test task-based recommendations
        proposals_models = registry.get_model_recommendations("proposals")
        assert any(model.name == "mistral-small" for model in proposals_models)
        
        # Test performance tier filtering
        premium_models = registry.get_model_recommendations("proposals", performance_tier="premium")
        assert all(model.performance_tier == "premium" for model in premium_models)
    
    def test_backend_name_mapping(self):
        """Test model name mapping between backends."""
        registry = ModelRegistry()
        
        # Test Docker name mapping
        docker_name = registry.get_docker_name("mistral")
        assert docker_name == "ai/mistral:7b-instruct-v0.3-q4_K_M"
        
        # Test Ollama name mapping
        ollama_name = registry.get_ollama_name("mistral")
        assert ollama_name == "mistral:7b-instruct-v0.3-q4_K_M"


class TestDockerModelService:
    """Test Docker Model Runner service integration."""
    
    @pytest.fixture
    def docker_service(self):
        """Create a Docker model service instance."""
        return DockerModelService()
    
    @pytest.mark.asyncio
    async def test_docker_availability_check(self, docker_service):
        """Test Docker availability checking."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful Docker check
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"Docker version", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            available = await docker_service.is_available()
            assert available == True
    
    @pytest.mark.asyncio  
    async def test_model_name_mapping(self, docker_service):
        """Test model name mapping to Docker registry."""
        # Test default mapping
        docker_name = docker_service._map_model_name("mistral")
        assert docker_name == "ai/mistral"
        
        # Test explicit mapping
        docker_name = docker_service._map_model_name("mistral:7b-instruct-v0.3-q4_K_M")
        assert docker_name == "ai/mistral:7b-instruct-v0.3-q4_K_M"
    
    @pytest.mark.asyncio
    async def test_model_pull_progress(self, docker_service):
        """Test model pull with progress tracking."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock pull process with progress output
            mock_process = AsyncMock()
            mock_process.stdout.readline.side_effect = [
                b"Downloading: 1.5GB / 4.0GB\n",
                b"Downloading: 4.0GB / 4.0GB\n", 
                b"Download completed successfully\n",
                b""  # EOF
            ]
            mock_process.wait.return_value = 0
            mock_subprocess.return_value = mock_process
            
            progress_updates = []
            async for progress in docker_service.pull_model("mistral"):
                progress_updates.append(progress)
            
            assert len(progress_updates) >= 3
            assert "completed successfully" in progress_updates[-1].status


class TestOllamaModelService:
    """Test Ollama model service integration."""
    
    @pytest.fixture
    def ollama_service(self):
        """Create an Ollama model service instance."""
        return OllamaModelService()
    
    @pytest.mark.asyncio
    async def test_ollama_availability_check(self, ollama_service):
        """Test Ollama availability checking."""
        with patch('cortex_engine.utils.ollama_utils.check_ollama_service') as mock_check:
            mock_check.return_value = (True, None)
            
            available = await ollama_service.is_available()
            assert available == True
    
    @pytest.mark.asyncio
    async def test_list_models(self, ollama_service):
        """Test listing available Ollama models."""
        mock_response_data = {
            "models": [
                {
                    "name": "mistral:7b-instruct-v0.3-q4_K_M",
                    "size": 4400000000
                },
                {
                    "name": "llava:latest", 
                    "size": 4500000000
                }
            ]
        }
        
        with patch.object(ollama_service, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value.status = 200
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value.json.return_value = mock_response_data
            
            models = await ollama_service.list_available_models()
            
            assert len(models) == 2
            assert models[0].name == "mistral"
            assert models[0].tag == "7b-instruct-v0.3-q4_K_M"
            assert models[0].backend == "ollama"


class TestHybridModelManager:
    """Test the hybrid model manager orchestration."""
    
    @pytest.fixture
    def hybrid_manager(self):
        """Create a hybrid model manager instance."""
        return HybridModelManager(strategy=DistributionStrategy.HYBRID_DOCKER_PREFERRED)
    
    @pytest.mark.asyncio
    async def test_strategy_selection(self, hybrid_manager):
        """Test that the manager selects appropriate strategies."""
        # Test Docker preference in production
        backend = hybrid_manager._get_optimal_backend("mistral", "production")
        assert backend == "docker_model_runner"
        
        # Test Ollama fallback
        backend = hybrid_manager._get_optimal_backend("mistral", "development")
        assert backend == "docker_model_runner"  # Still preferred for this strategy
    
    @pytest.mark.asyncio
    async def test_backend_availability_caching(self, hybrid_manager):
        """Test that backend availability is cached appropriately."""
        with patch.object(hybrid_manager.docker_service, 'is_available', return_value=True) as mock_docker:
            with patch.object(hybrid_manager.ollama_service, 'is_available', return_value=True) as mock_ollama:
                
                # First call should check both services
                availability1 = await hybrid_manager._check_backend_availability()
                assert mock_docker.call_count == 1
                assert mock_ollama.call_count == 1
                
                # Second call should use cache
                availability2 = await hybrid_manager._check_backend_availability()
                assert mock_docker.call_count == 1  # No additional calls
                assert mock_ollama.call_count == 1
                
                assert availability1 == availability2
    
    @pytest.mark.asyncio
    async def test_optimal_service_selection(self, hybrid_manager):
        """Test optimal service selection with fallback."""
        with patch.object(hybrid_manager, '_check_backend_availability') as mock_availability:
            mock_availability.return_value = {
                "docker_model_runner": True,
                "ollama": True
            }
            
            with patch.object(hybrid_manager, '_get_backend_service') as mock_get_service:
                mock_docker_service = Mock()
                mock_ollama_service = Mock()
                
                def get_service_side_effect(backend_name):
                    if backend_name == "docker_model_runner":
                        return mock_docker_service
                    elif backend_name == "ollama":
                        return mock_ollama_service
                    return None
                
                mock_get_service.side_effect = get_service_side_effect
                
                # Should prefer Docker Model Runner
                service = await hybrid_manager.get_optimal_service_for_model("mistral")
                assert service == mock_docker_service
    
    @pytest.mark.asyncio
    async def test_system_status(self, hybrid_manager):
        """Test system status reporting."""
        with patch.object(hybrid_manager, '_check_backend_availability') as mock_availability:
            mock_availability.return_value = {
                "docker_model_runner": True,
                "ollama": False
            }
            
            with patch.object(hybrid_manager.docker_service, 'list_available_models') as mock_list:
                mock_list.return_value = [
                    ModelInfo("mistral", "7b", 4.4, ModelStatus.AVAILABLE, "docker_model_runner")
                ]
                
                status = await hybrid_manager.get_system_status()
                
                assert status["strategy"] == "hybrid_docker_preferred"
                assert "docker_model_runner" in status["available_backends"]
                assert "ollama" not in status["available_backends"]
                assert status["total_models"] == 1


class TestModelMigration:
    """Test model migration utilities."""
    
    @pytest.fixture
    def migration_manager(self):
        """Create migration manager with mocked hybrid manager."""
        mock_hybrid = AsyncMock()
        return ModelMigrationManager(mock_hybrid)
    
    @pytest.mark.asyncio
    async def test_migration_plan_creation(self, migration_manager):
        """Test creation of migration plans."""
        # Mock source service with available models
        mock_source_service = AsyncMock()
        mock_source_service.list_available_models.return_value = [
            ModelInfo("mistral", "7b", 4.4, ModelStatus.AVAILABLE, "ollama")
        ]
        
        # Mock target service
        mock_target_service = AsyncMock()
        mock_target_service.is_model_available.return_value = False
        
        with patch.object(migration_manager.hybrid_manager, '_get_backend_service') as mock_get_service:
            mock_get_service.side_effect = [mock_source_service, mock_target_service]
            
            plan = await migration_manager.create_migration_plan("ollama", "docker_model_runner")
            
            assert plan.source_backend == "ollama"
            assert plan.target_backend == "docker_model_runner"
            assert len(plan.models_to_migrate) == 1
            assert plan.estimated_size_gb == 4.4
    
    def test_model_name_mapping(self, migration_manager):
        """Test model name mapping between backends."""
        # Ollama to Docker
        docker_name = migration_manager._get_target_model_name(
            "mistral:7b-instruct-v0.3-q4_K_M", "ollama", "docker_model_runner"
        )
        assert docker_name == "ai/mistral:7b-instruct-v0.3-q4_K_M"
        
        # Docker to Ollama 
        ollama_name = migration_manager._get_target_model_name(
            "ai/mistral:7b-instruct", "docker_model_runner", "ollama"
        )
        assert ollama_name == "mistral:7b-instruct"
    
    @pytest.mark.asyncio
    async def test_migration_validation(self, migration_manager):
        """Test migration compatibility validation."""
        with patch.object(migration_manager.hybrid_manager, '_check_backend_availability', 
                         new_callable=AsyncMock) as mock_availability:
            mock_availability.return_value = {
                "ollama": True,
                "docker_model_runner": False
            }
            
            validation = await migration_manager.validate_migration_compatibility("ollama", "docker_model_runner")
            
            assert validation["source_available"] == True
            assert validation["target_available"] == False
            assert validation["compatible"] == False
            assert "Target backend docker_model_runner is not available" in validation["issues"]


class TestIntegration:
    """Integration tests for the complete hybrid system."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_model_usage(self):
        """Test complete model usage workflow."""
        # This would be a real integration test that requires Docker/Ollama
        # Skip in CI unless integration testing environment is available
        pytest.skip("Integration test requires Docker/Ollama environment")
        
        manager = HybridModelManager()
        
        try:
            # Check system availability
            backends = await manager.get_available_backends()
            if not backends:
                pytest.skip("No model backends available")
            
            # List available models
            models = await manager.list_all_available_models()
            if not models:
                pytest.skip("No models available for testing")
            
            # Test inference with first available model
            test_model = models[0].full_name
            inference_works = await manager.test_model_inference(test_model, "Hello test")
            assert inference_works
            
            # Get performance metrics
            metrics = await manager.get_performance_metrics(test_model)
            assert "backend" in metrics
            assert "performance_tier" in metrics
            
        finally:
            await manager.close()


# Test configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_model_services.py -v
    pytest.main([__file__, "-v"])