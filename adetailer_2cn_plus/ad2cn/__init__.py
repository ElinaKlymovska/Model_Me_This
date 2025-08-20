"""
ADetailer 2CN Plus - Advanced face detection and alignment pipeline for A1111.
"""

__version__ = "0.1.0"
__author__ = "ADetailer 2CN Team"

from .config import Config
from .pipeline.detect import DetectionPipeline
from .pipeline.a_pass import APassPipeline
from .pipeline.b_pass import BPassPipeline

__all__ = [
    "Config",
    "DetectionPipeline", 
    "APassPipeline",
    "BPassPipeline",
]
