"""
Tests for configuration module.
"""

import pytest
from pathlib import Path
from ad2cn.config import Config, DetectorConfig, SearchConfig, PipelineConfig


class TestDetectorConfig:
    """Test DetectorConfig class."""
    
    def test_valid_detector_names(self):
        """Test valid detector names."""
        valid_names = ['blazeface', 'retinaface', 'mtcnn', 'scrfd']
        
        for name in valid_names:
            config = DetectorConfig(name=name)
            assert config.name == name
    
    def test_invalid_detector_name(self):
        """Test invalid detector name raises error."""
        with pytest.raises(ValueError, match="Detector must be one of"):
            DetectorConfig(name="invalid_detector")
    
    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation."""
        # Valid thresholds
        config = DetectorConfig(name="blazeface", confidence_threshold=0.5)
        assert config.confidence_threshold == 0.5
        
        # Invalid thresholds
        with pytest.raises(ValueError):
            DetectorConfig(name="blazeface", confidence_threshold=1.5)
        
        with pytest.raises(ValueError):
            DetectorConfig(name="blazeface", confidence_threshold=-0.1)


class TestSearchConfig:
    """Test SearchConfig class."""
    
    def test_valid_strategies(self):
        """Test valid search strategies."""
        valid_strategies = ['sliding_window', 'multi_scale']
        
        for strategy in valid_strategies:
            config = SearchConfig(strategy=strategy)
            assert config.strategy == strategy
    
    def test_invalid_strategy(self):
        """Test invalid strategy raises error."""
        with pytest.raises(ValueError, match="Strategy must be one of"):
            SearchConfig(strategy="invalid_strategy")
    
    def test_window_size_validation(self):
        """Test window size validation."""
        # Valid sizes
        config = SearchConfig(window_size=512)
        assert config.window_size == 512
        
        # Invalid sizes
        with pytest.raises(ValueError):
            SearchConfig(window_size=32)  # Too small


class TestPipelineConfig:
    """Test PipelineConfig class."""
    
    def test_valid_cascade_order(self):
        """Test valid cascade order."""
        detectors = [
            DetectorConfig(name="blazeface"),
            DetectorConfig(name="retinaface")
        ]
        
        config = PipelineConfig(
            detectors=detectors,
            cascade_order=["blazeface", "retinaface"]
        )
        
        assert config.cascade_order == ["blazeface", "retinaface"]
    
    def test_invalid_cascade_order(self):
        """Test invalid cascade order raises error."""
        detectors = [DetectorConfig(name="blazeface")]
        
        with pytest.raises(ValueError, match="Cascade order contains unknown detector"):
            PipelineConfig(
                detectors=detectors,
                cascade_order=["blazeface", "unknown_detector"]
            )


class TestConfig:
    """Test main Config class."""
    
    def test_minimal_config(self):
        """Test minimal valid configuration."""
        minimal_config = {
            "pipeline": {
                "detectors": [{"name": "blazeface"}]
            }
        }
        
        config = Config(**minimal_config)
        assert config.pipeline.detectors[0].name == "blazeface"
    
    def test_full_config(self):
        """Test full configuration."""
        full_config = {
            "pipeline": {
                "detectors": [
                    {"name": "blazeface", "confidence_threshold": 0.5},
                    {"name": "retinaface", "confidence_threshold": 0.7}
                ],
                "search": {
                    "strategy": "sliding_window",
                    "window_size": 512
                },
                "enable_cascade": True,
                "cascade_order": ["blazeface", "retinaface"]
            },
            "a_pass": {"enabled": True, "batch_size": 4},
            "b_pass": {"enabled": True, "batch_size": 4},
            "num_workers": 4,
            "use_gpu": True,
            "log_level": "INFO"
        }
        
        config = Config(**full_config)
        assert len(config.pipeline.detectors) == 2
        assert config.pipeline.search.strategy == "sliding_window"
        assert config.a_pass.enabled is True
        assert config.b_pass.enabled is True
    
    def test_log_level_validation(self):
        """Test log level validation."""
        config_data = {
            "pipeline": {"detectors": [{"name": "blazeface"}]},
            "log_level": "INFO"
        }
        
        config = Config(**config_data)
        assert config.log_level == "INFO"
        
        # Test invalid log level
        with pytest.raises(ValueError, match="Log level must be one of"):
            Config(**{**config_data, "log_level": "INVALID"})
    
    def test_gpu_memory_fraction_validation(self):
        """Test GPU memory fraction validation."""
        config_data = {
            "pipeline": {"detectors": [{"name": "blazeface"}]},
            "gpu_memory_fraction": 0.8
        }
        
        config = Config(**config_data)
        assert config.gpu_memory_fraction == 0.8
        
        # Test invalid values
        with pytest.raises(ValueError):
            Config(**{**config_data, "gpu_memory_fraction": 1.5})
        
        with pytest.raises(ValueError):
            Config(**{**config_data, "gpu_memory_fraction": 0.05})
