"""
Simple OpenCV face detector using Haar cascades.
"""

import numpy as np
from typing import List, Dict, Any
import cv2
from .base import FaceDetector


class OpenCVDetector(FaceDetector):
    """OpenCV Haar cascade face detector."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenCV detector.
        
        Args:
            config: Detector configuration
        """
        super().__init__(config)
        self.scale_factor = config.get('scale_factor', 1.1)
        self.min_neighbors = config.get('min_neighbors', 5)
        self.min_size = config.get('min_size', (30, 30))
        self.cascade = None
        self._initialized = False
    
    def load_model(self) -> None:
        """Load OpenCV Haar cascade model."""
        try:
            # Use built-in frontal face cascade
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            if self.cascade.empty():
                raise RuntimeError("Failed to load Haar cascade classifier")
            
            self._initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenCV face detector: {e}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using OpenCV Haar cascades.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            List of detections
        """
        if not self._initialized:
            self.load_model()
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        detections = []
        for (x, y, w, h) in faces:
            # Calculate confidence (simple heuristic based on face size)
            confidence = min(0.9, max(0.5, (w * h) / (image.shape[0] * image.shape[1]) * 10))
            
            if confidence > self.confidence_threshold:
                detections.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': float(confidence),
                    'landmarks': None,
                    'detector': 'opencv'
                })
        
        return detections