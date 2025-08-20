"""
RetinaFace detector implementation.
"""

import numpy as np
from typing import List, Dict, Any
import cv2
from .base import FaceDetector


class RetinaFaceDetector(FaceDetector):
    """RetinaFace face detector implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize RetinaFace detector.
        
        Args:
            config: Detector configuration
        """
        super().__init__(config)
        self.input_size = config.get('input_size', 640)
        self.model = None
        self._initialized = False
    
    def load_model(self) -> None:
        """Load RetinaFace model."""
        try:
            # Try to load ONNX model first
            import onnxruntime as ort
            
            if self.model_path and self.model_path.exists():
                self.model = ort.InferenceSession(str(self.model_path))
            else:
                # Load default model or download
                self.model = self._load_default_model()
            
            self._initialized = True
            
        except ImportError:
            raise ImportError("onnxruntime is required for RetinaFace. Install with: pip install onnxruntime-gpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load RetinaFace model: {e}")
    
    def _load_default_model(self):
        """Load default RetinaFace model."""
        # This would typically download or use a bundled model
        # For now, we'll require a model path
        raise ValueError("RetinaFace requires a model_path in configuration")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for RetinaFace.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
        
        # Resize to input size
        resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # Normalize to [0, 1]
        resized = np.clip(resized, 0, 1)
        
        # Add batch dimension
        resized = np.expand_dims(resized, axis=0)
        
        return resized
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using RetinaFace.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of detections
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Run inference
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: image})
        
        # Parse outputs
        detections = self._parse_outputs(outputs, image.shape[1:3])
        
        return detections
    
    def _parse_outputs(self, outputs, original_shape):
        """Parse model outputs to get detections.
        
        Args:
            outputs: Model outputs
            original_shape: Original image shape (H, W)
            
        Returns:
            List of detections
        """
        detections = []
        
        # RetinaFace typically outputs: [boxes, scores, landmarks]
        if len(outputs) >= 2:
            boxes = outputs[0]  # Detection boxes
            scores = outputs[1]  # Confidence scores
            landmarks = outputs[2] if len(outputs) > 2 else None
            
            for i in range(len(scores)):
                if scores[i] > self.confidence_threshold:
                    # Convert normalized coordinates to pixel coordinates
                    x1, y1, x2, y2 = boxes[i]
                    x1 = int(x1 * original_shape[1])
                    y1 = int(y1 * original_shape[0])
                    x2 = int(x2 * original_shape[1])
                    y2 = int(y2 * original_shape[0])
                    
                    # Extract landmarks if available
                    face_landmarks = None
                    if landmarks is not None:
                        face_landmarks = landmarks[i].reshape(-1, 2)
                        # Convert to pixel coordinates
                        face_landmarks[:, 0] *= original_shape[1]
                        face_landmarks[:, 1] *= original_shape[0]
                        face_landmarks = face_landmarks.astype(np.int32).tolist()
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(scores[i]),
                        'landmarks': face_landmarks,
                        'detector': 'retinaface'
                    })
        
        return detections
