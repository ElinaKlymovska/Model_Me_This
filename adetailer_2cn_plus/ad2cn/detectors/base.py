"""
Abstract base class for face detectors.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path


class FaceDetector(ABC):
    """Abstract base class for face detectors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize detector with configuration.
        
        Args:
            config: Detector configuration dictionary
        """
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.3)
        self.max_faces = config.get('max_faces', 100)
        self.model_path = config.get('model_path')
        self.model = None
        self._initialized = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in the image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - confidence: Detection confidence score
            - landmarks: Optional facial landmarks
            - detector: Detector name
        """
        pass
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for detection.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Default preprocessing: ensure RGB format and normalize
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB if needed
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
        return image
    
    def postprocess(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Postprocess detections.
        
        Args:
            detections: Raw detections
            
        Returns:
            Postprocessed detections
        """
        # Filter by confidence
        filtered = [d for d in detections if d['confidence'] >= self.confidence_threshold]
        
        # Sort by confidence (descending)
        filtered.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit number of faces
        if len(filtered) > self.max_faces:
            filtered = filtered[:self.max_faces]
        
        # Add detector name
        for det in filtered:
            det['detector'] = self.__class__.__name__.lower()
        
        return filtered
    
    def is_initialized(self) -> bool:
        """Check if detector is properly initialized."""
        return self._initialized
    
    def __call__(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Convenience method to run detection pipeline.
        
        Args:
            image: Input image
            
        Returns:
            List of detections
        """
        if not self._initialized:
            self.load_model()
        
        # Preprocess
        processed_image = self.preprocess(image)
        
        # Detect
        raw_detections = self.detect(processed_image)
        
        # Postprocess
        final_detections = self.postprocess(raw_detections)
        
        return final_detections
