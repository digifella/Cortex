"""
Unit Tests for Model Availability Checker
Version: 1.0.0
Purpose: Test AI model availability checking and validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from cortex_engine.utils.model_checker import ModelAvailabilityChecker


class TestModelCheckerInitialization:
    """Test ModelAvailabilityChecker initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        checker = ModelAvailabilityChecker()
        assert checker is not None

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_client_initialization_success(self, mock_client):
        """Test successful Ollama client initialization."""
        mock_client.return_value = Mock()
        checker = ModelAvailabilityChecker()
        assert checker.client is not None

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_client_initialization_failure(self, mock_client):
        """Test handling of Ollama client initialization failure."""
        mock_client.side_effect = Exception("Connection failed")
        checker = ModelAvailabilityChecker()
        assert checker.client is None


class TestOllamaServiceCheck:
    """Test Ollama service availability checking."""

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_service_running(self, mock_client):
        """Test detection of running Ollama service."""
        mock_instance = Mock()
        mock_instance.list.return_value = {'models': []}
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        is_running, message = checker.check_ollama_service()

        assert is_running is True
        assert "running" in message.lower()

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_service_not_running(self, mock_client):
        """Test detection of non-running Ollama service."""
        mock_instance = Mock()
        mock_instance.list.side_effect = Exception("Connection refused")
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        is_running, message = checker.check_ollama_service()

        assert is_running is False
        assert "not available" in message.lower()

    def test_service_check_without_client(self):
        """Test service check when client is not initialized."""
        checker = ModelAvailabilityChecker()
        checker.client = None

        is_running, message = checker.check_ollama_service()

        assert is_running is False
        assert "not be initialized" in message


class TestModelAvailability:
    """Test individual model availability checking."""

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_model_available_exact_match(self, mock_client):
        """Test detection of available model with exact name match."""
        mock_instance = Mock()
        mock_instance.list.return_value = {
            'models': [
                {'name': 'mistral:latest'},
                {'name': 'llava:7b'},
            ]
        }
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        is_available, message = checker.check_model_availability('mistral:latest')

        assert is_available is True
        assert 'available' in message.lower()

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_model_available_base_match(self, mock_client):
        """Test detection of available model with base name match."""
        mock_instance = Mock()
        mock_instance.list.return_value = {
            'models': [
                {'name': 'mistral:latest'},
            ]
        }
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        is_available, message = checker.check_model_availability('mistral')

        assert is_available is True

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_model_not_available(self, mock_client):
        """Test detection of unavailable model."""
        mock_instance = Mock()
        mock_instance.list.return_value = {
            'models': [
                {'name': 'mistral:latest'},
            ]
        }
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        is_available, message = checker.check_model_availability('llama2:70b')

        assert is_available is False
        assert 'not found' in message.lower()

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_empty_model_list(self, mock_client):
        """Test handling of empty model list."""
        mock_instance = Mock()
        mock_instance.list.return_value = {'models': []}
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        is_available, message = checker.check_model_availability('any-model')

        assert is_available is False


class TestIngestionRequirements:
    """Test ingestion requirements checking."""

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_all_requirements_met(self, mock_client):
        """Test when all ingestion requirements are met."""
        mock_instance = Mock()
        mock_instance.list.return_value = {
            'models': [
                {'name': 'mistral:latest'},
                {'name': 'llava:7b'},
            ]
        }
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        result = checker.check_ingestion_requirements(include_images=True)

        assert result['ollama_running'] is True
        assert result['kb_model_available'] is True
        assert result['vlm_model_available'] is True
        assert result['can_proceed'] is True
        assert len(result['missing_models']) == 0

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_missing_kb_model(self, mock_client):
        """Test detection of missing KB model."""
        mock_instance = Mock()
        mock_instance.list.return_value = {
            'models': [
                {'name': 'llava:7b'},
            ]
        }
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        result = checker.check_ingestion_requirements(include_images=True)

        assert result['kb_model_available'] is False
        assert result['can_proceed'] is False
        assert len(result['missing_models']) > 0

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_missing_vlm_model(self, mock_client):
        """Test detection of missing VLM model when images enabled."""
        mock_instance = Mock()
        mock_instance.list.return_value = {
            'models': [
                {'name': 'mistral:latest'},
            ]
        }
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        result = checker.check_ingestion_requirements(include_images=True)

        assert result['vlm_model_available'] is False
        assert result['can_proceed'] is False

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_vlm_not_required_when_images_disabled(self, mock_client):
        """Test that VLM is not required when image processing is disabled."""
        mock_instance = Mock()
        mock_instance.list.return_value = {
            'models': [
                {'name': 'mistral:latest'},
            ]
        }
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        result = checker.check_ingestion_requirements(include_images=False)

        assert result['vlm_model_available'] is True  # Not required, so marked True
        assert result['can_proceed'] is True

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_ollama_not_running(self, mock_client):
        """Test ingestion requirements when Ollama is not running."""
        mock_client.side_effect = Exception("Connection failed")

        checker = ModelAvailabilityChecker()
        result = checker.check_ingestion_requirements()

        assert result['ollama_running'] is False
        assert result['can_proceed'] is False
        assert len(result['warnings']) > 0
        assert len(result['recommended_actions']) > 0


class TestResearchRequirements:
    """Test research operation requirements checking."""

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_research_requirements_met(self, mock_client):
        """Test when research requirements are met."""
        mock_instance = Mock()
        mock_instance.list.return_value = {
            'models': [
                {'name': 'mistral:latest'},
            ]
        }
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        result = checker.check_research_requirements()

        assert result['ollama_running'] is True
        assert result['local_research_available'] is True

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_research_ollama_not_running(self, mock_client):
        """Test research requirements when Ollama not running."""
        mock_client.side_effect = Exception("Connection failed")

        checker = ModelAvailabilityChecker()
        result = checker.check_research_requirements()

        assert result['ollama_running'] is False
        assert len(result['warnings']) > 0


class TestGetAvailableModels:
    """Test retrieval of available models list."""

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_get_model_list(self, mock_client):
        """Test getting list of available models."""
        mock_instance = Mock()
        mock_instance.list.return_value = {
            'models': [
                {'name': 'mistral:latest'},
                {'name': 'llava:7b'},
                {'name': 'gemma:2b'},
            ]
        }
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        models = checker.get_available_models()

        assert len(models) == 3
        assert 'mistral:latest' in models
        assert 'llava:7b' in models

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_get_model_list_error(self, mock_client):
        """Test handling of errors when getting model list."""
        mock_instance = Mock()
        mock_instance.list.side_effect = Exception("API error")
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        models = checker.get_available_models()

        assert models == []

    def test_get_models_without_client(self):
        """Test getting models when client is not initialized."""
        checker = ModelAvailabilityChecker()
        checker.client = None

        models = checker.get_available_models()
        assert models == []


# ============================================================================
# Integration-style tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.requires_ollama
class TestModelCheckerIntegration:
    """Integration tests requiring actual Ollama service."""

    def test_real_ollama_connection(self):
        """Test connection to real Ollama service."""
        # This test will be skipped unless Ollama is running
        checker = ModelAvailabilityChecker()
        is_running, message = checker.check_ollama_service()

        if is_running:
            # If Ollama is running, we should get a valid model list
            models = checker.get_available_models()
            assert isinstance(models, list)
        else:
            # If not running, that's also a valid state
            assert "not available" in message.lower() or "not be initialized" in message


# ============================================================================
# Error handling and edge cases
# ============================================================================

class TestErrorHandling:
    """Test error handling in model checker."""

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_malformed_model_list_response(self, mock_client):
        """Test handling of malformed model list response."""
        mock_instance = Mock()
        mock_instance.list.return_value = {'wrong_key': []}
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        models = checker.get_available_models()

        # Should return empty list instead of crashing
        assert models == []

    @patch('cortex_engine.utils.model_checker.ollama.Client')
    def test_none_model_name(self, mock_client):
        """Test checking availability of None model name."""
        mock_instance = Mock()
        mock_instance.list.return_value = {'models': [{'name': 'mistral:latest'}]}
        mock_client.return_value = mock_instance

        checker = ModelAvailabilityChecker()
        # Should handle gracefully
        try:
            is_available, message = checker.check_model_availability(None)
            assert is_available is False
        except (TypeError, AttributeError):
            # Either way is acceptable
            pass
