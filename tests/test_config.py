"""
Tests for the configuration system
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add the project root to the path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    OllamaConfig,
    ChromaConfig,
    EmbeddingConfig,
    FastAPIConfig,
    InterviewConfig,
    ReportConfig,
    AppConfig,
    validate_config
)

class TestOllamaConfig:
    """Test Ollama configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = OllamaConfig()
        assert config.base_url == "http://localhost:11434"
        assert config.model_name == "mistral:7b"
        assert config.timeout == 60
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert 0.0 <= config.temperature <= 2.0
        assert 0.0 <= config.top_p <= 1.0

    @patch.dict(os.environ, {
        'OLLAMA_BASE_URL': 'http://custom:11434',
        'OLLAMA_MODEL': 'llama2:7b',
        'OLLAMA_TEMPERATURE': '0.5'
    })
    def test_environment_overrides(self):
        """Test that environment variables override defaults."""
        config = OllamaConfig()
        assert config.base_url == "http://custom:11434"
        assert config.model_name == "llama2:7b"
        assert config.temperature == 0.5

class TestChromaConfig:
    """Test ChromaDB configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ChromaConfig()
        assert config.db_path == "./data/vector_db"
        assert config.collection_name == "interview_knowledge"

    @patch.dict(os.environ, {
        'CHROMA_DB_PATH': '/custom/path',
        'CHROMA_COLLECTION_NAME': 'custom_collection'
    })
    def test_environment_overrides(self):
        """Test that environment variables override defaults."""
        config = ChromaConfig()
        assert config.db_path == "/custom/path"
        assert config.collection_name == "custom_collection"

class TestEmbeddingConfig:
    """Test embedding configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.device == "cpu"

class TestFastAPIConfig:
    """Test FastAPI configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = FastAPIConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.reload is True
        assert config.log_level == "info"
        assert isinstance(config.cors_origins, list)

class TestInterviewConfig:
    """Test interview configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = InterviewConfig()
        assert config.max_questions_per_session == 10
        assert config.default_question_count == 5
        assert isinstance(config.available_roles, list)
        assert isinstance(config.available_levels, list)
        assert "software_engineer" in config.available_roles
        assert "data_scientist" in config.available_roles
        assert "junior" in config.available_levels
        assert "senior" in config.available_levels

class TestReportConfig:
    """Test report configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ReportConfig()
        assert config.reports_dir == "./reports"
        assert config.include_charts is True
        assert isinstance(config.formats, list)

class TestAppConfig:
    """Test application configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = AppConfig()
        assert config.log_level == "INFO"
        assert config.debug is False
        assert config.development_mode is True

class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test that default configuration is valid."""
        # This should pass with default values
        assert validate_config() is True

    @patch.dict(os.environ, {'OLLAMA_TEMPERATURE': '3.0'})
    def test_invalid_temperature(self):
        """Test validation fails with invalid temperature."""
        # Temperature out of range should fail validation
        assert validate_config() is False

    @patch.dict(os.environ, {'OLLAMA_TOP_P': '1.5'})
    def test_invalid_top_p(self):
        """Test validation fails with invalid top_p."""
        # top_p out of range should fail validation
        assert validate_config() is False

    @patch.dict(os.environ, {'OLLAMA_MAX_TOKENS': '-1'})
    def test_invalid_max_tokens(self):
        """Test validation fails with invalid max_tokens."""
        # Negative max_tokens should fail validation
        assert validate_config() is False

    @patch.dict(os.environ, {
        'DEFAULT_QUESTION_COUNT': '15',
        'MAX_QUESTIONS_PER_SESSION': '10'
    })
    def test_invalid_question_counts(self):
        """Test validation fails when default > max questions."""
        # Default questions > max questions should fail validation
        assert validate_config() is False

    @patch.dict(os.environ, {'API_PORT': '99999'})
    def test_invalid_port(self):
        """Test validation fails with invalid port."""
        # Port out of valid range should fail validation
        assert validate_config() is False

class TestDirectoryCreation:
    """Test that validation creates necessary directories."""
    
    def test_directory_creation(self):
        """Test that validate_config creates required directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set temporary paths
            with patch.dict(os.environ, {
                'CHROMA_DB_PATH': f'{temp_dir}/chroma',
                'REPORTS_DIR': f'{temp_dir}/reports'
            }):
                # Run validation
                result = validate_config()
                
                # Check that directories were created
                assert Path(f'{temp_dir}/chroma').exists()
                assert Path(f'{temp_dir}/reports').exists()
                assert result is True

if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])