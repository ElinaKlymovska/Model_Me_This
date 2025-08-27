"""Simple single face detector."""

import numpy as np
from typing import List, Dict, Any
from .base import FaceDetector
from .retinaface import RetinaFaceDetector


class SingleFaceDetector(FaceDetector):
    """Detector optimized for single face scenarios."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.detector = RetinaFaceDetector(config)
        
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect single face in image."""
        detections = self.detector.detect(image)
        
        if not detections:
            return []
            
        # Return only the highest confidence detection
        best_detection = max(detections, key=lambda d: d.get('confidence', 0))
        return [best_detection]