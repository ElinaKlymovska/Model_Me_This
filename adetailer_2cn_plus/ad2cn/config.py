"""
Configuration management using Pydantic BaseModel.
"""

from pathlib import Path
from typing import Optional, List, Union
from pydantic import BaseModel, Field, validator


class DetectorConfig(BaseModel):
    """Configuration for face detectors."""
    name: str = Field(..., description="Detector name: blazeface, retinaface, mtcnn, scrfd, opencv")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Detection confidence threshold")
    nms_threshold: float = Field(0.3, ge=0.0, le=1.0, description="NMS threshold")
    max_faces: int = Field(100, ge=1, description="Maximum number of faces to detect")
    model_path: Optional[Path] = Field(None, description="Path to detector model file")
    
    @validator('name')
    def validate_detector_name(cls, v):
        valid_names = ['blazeface', 'retinaface', 'mtcnn', 'scrfd', 'opencv', 'mediapipe']
        if v not in valid_names:
            raise ValueError(f'Detector must be one of: {valid_names}')
        return v


class SearchConfig(BaseModel):
    """Configuration for face search strategies."""
    strategy: str = Field('sliding_window', description="Search strategy: sliding_window, multi_scale")
    window_size: int = Field(512, ge=64, description="Sliding window size")
    stride: int = Field(256, ge=32, description="Window stride")
    scale_factors: List[float] = Field([1.0, 0.75, 0.5], description="Multi-scale factors")
    
    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ['sliding_window', 'multi_scale']
        if v not in valid_strategies:
            raise ValueError(f'Strategy must be one of: {valid_strategies}')
        return v


class PipelineConfig(BaseModel):
    """Configuration for detection pipeline."""
    detectors: List[DetectorConfig] = Field(..., description="List of detector configurations")
    search: SearchConfig = Field(default_factory=SearchConfig, description="Search configuration")
    enable_cascade: bool = Field(True, description="Enable cascade detection")
    cascade_order: List[str] = Field(['blazeface', 'retinaface'], description="Cascade detection order")
    
    @validator('cascade_order')
    def validate_cascade_order(cls, v, values):
        if 'detectors' in values:
            detector_names = [d.name for d in values['detectors']]
            for name in v:
                if name not in detector_names:
                    raise ValueError(f'Cascade order contains unknown detector: {name}')
        return v


class APassConfig(BaseModel):
    """Configuration for A-Pass pipeline."""
    enabled: bool = Field(True, description="Enable A-Pass processing")
    model_path: Optional[Path] = Field(None, description="Path to A-Pass model")
    batch_size: int = Field(4, ge=1, description="Batch size for processing")
    device: str = Field('auto', description="Device for processing: auto, cpu, cuda")


class BPassConfig(BaseModel):
    """Configuration for B-Pass pipeline."""
    enabled: bool = Field(True, description="Enable B-Pass processing")
    model_path: Optional[Path] = Field(None, description="Path to B-Pass model")
    batch_size: int = Field(4, ge=1, description="Batch size for processing")
    device: str = Field('auto', description="Device for processing: auto, cpu, cuda")


class PersonProfileConfig(BaseModel):
    """Configuration for person-specific settings."""
    contour_strength: float = Field(0.7, ge=0.0, le=1.0, description="Contouring strength")
    expression_strength: float = Field(0.5, ge=0.0, le=1.0, description="Expression lines strength")


class ContouringConfig(BaseModel):
    """Configuration for facial contouring."""
    enabled: bool = Field(True, description="Enable contouring")
    intensity: float = Field(0.7, ge=0.0, le=1.0, description="Overall contouring intensity")
    blend_strength: float = Field(0.5, ge=0.0, le=1.0, description="Blending strength with original")


class ExpressionLinesConfig(BaseModel):
    """Configuration for expression lines."""
    enabled: bool = Field(True, description="Enable expression lines")
    neutral_intensity: float = Field(0.3, ge=0.0, le=1.0, description="Neutral expression intensity")
    smile_intensity: float = Field(0.6, ge=0.0, le=1.0, description="Smiling expression intensity")


class FacialEnhancementConfig(BaseModel):
    """Configuration for facial enhancement system."""
    contouring: ContouringConfig = Field(default_factory=ContouringConfig)
    expression_lines: ExpressionLinesConfig = Field(default_factory=ExpressionLinesConfig)
    person_profiles: dict = Field(default_factory=lambda: {
        'massy': {'contour_strength': 0.8, 'expression_strength': 0.5},
        'orbi': {'contour_strength': 0.7, 'expression_strength': 0.6},
        'yana': {'contour_strength': 0.75, 'expression_strength': 0.55}
    }, description="Person-specific profile settings")


class Config(BaseModel):
    """Main configuration class."""
    pipeline: PipelineConfig
    a_pass: APassConfig = Field(default_factory=APassConfig)
    b_pass: BPassConfig = Field(default_factory=BPassConfig)
    facial_enhancement: FacialEnhancementConfig = Field(default_factory=FacialEnhancementConfig)
    
    # I/O settings
    input_dir: Optional[Path] = Field(None, description="Input directory for batch processing")
    output_dir: Optional[Path] = Field(None, description="Output directory for results")
    
    # Performance settings
    num_workers: int = Field(4, ge=1, description="Number of worker processes")
    use_gpu: bool = Field(True, description="Enable GPU acceleration")
    gpu_memory_fraction: float = Field(0.8, ge=0.1, le=1.0, description="GPU memory fraction")
    
    # Logging
    log_level: str = Field('INFO', description="Logging level")
    log_file: Optional[Path] = Field(None, description="Log file path")
    
    # A1111 integration
    a1111_base_url: Optional[str] = Field(None, description="A1111 base URL")
    a1111_api_key: Optional[str] = Field(None, description="A1111 API key")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()
    
    class Config:
        validate_assignment = True
        extra = "forbid"
