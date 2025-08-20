"""
MTCNN detector implementation.
"""

import numpy as np
from typing import List, Dict, Any
import cv2
from .base import FaceDetector


class MTCNNDetector(FaceDetector):
    """MTCNN face detector implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MTCNN detector.
        
        Args:
            config: Detector configuration
        """
        super().__init__(config)
        self.min_face_size = config.get('min_face_size', 20)
        self.factor = config.get('factor', 0.709)
        self.thresholds = config.get('thresholds', [0.6, 0.7, 0.7])
        self.model = None
        self._initialized = False
    
    def load_model(self) -> None:
        """Load MTCNN model."""
        try:
            from facenet_pytorch import MTCNN
            
            # Initialize MTCNN with configuration
            self.model = MTCNN(
                min_face_size=self.min_face_size,
                factor=self.factor,
                thresholds=self.thresholds,
                device='cuda' if self.config.get('use_gpu', True) else 'cpu'
            )
            
            self._initialized = True
            
        except ImportError:
            raise ImportError("facenet_pytorch is required for MTCNN. Install with: pip install facenet-pytorch")
        except Exception as e:
            raise RuntimeError(f"Failed to load MTCNN model: {e}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for MTCNN.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Preprocessed image
        """
        # MTCNN expects RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB if needed
            if image.dtype == np.uint8:
                # Convert to PIL Image for MTCNN
                from PIL import Image
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                # Float array, ensure RGB
                image = Image.fromarray(image)
        else:
            # Grayscale, convert to RGB
            from PIL import Image
            image = Image.fromarray(image).convert('RGB')
        
        return image
    
    def detect(self, image) -> List[Dict[str, Any]]:
        """Detect faces using MTCNN.
        
        Args:
            image: Preprocessed image (PIL Image)
            
        Returns:
            List of detections
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Run detection
        boxes, probs, landmarks = self.model.detect(image, landmarks=True)
        
        # Convert to our format
        detections = []
        
        if boxes is not None:
            for i in range(len(boxes)):
                if probs[i] > self.confidence_threshold:
                    # MTCNN returns [x1, y1, x2, y2]
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    
                    # Extract landmarks if available
                    face_landmarks = None
                    if landmarks is not None:
                        face_landmarks = landmarks[i].astype(np.int32).tolist()
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(probs[i]),
                        'landmarks': face_landmarks,
                        'detector': 'mtcnn'
                    })
        
        return detections
