"""Simple detection pipeline."""

import numpy as np
from typing import List, Dict, Any
import cv2
from ..detectors.single_face_detector import SingleFaceDetector
from ..detectors.retinaface import RetinaFaceDetector
from ..detectors.blazeface import BlazeFaceDetector
from ..detectors.opencv_detector import OpenCVDetector
from ..detectors.mediapipe_detector import MediaPipeDetector


class DetectionPipeline:
    """Main face detection pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detectors = self._setup_detectors()
        
    def _setup_detectors(self) -> Dict[str, Any]:
        """Setup detectors from config."""
        detectors = {}
        
        # Get detectors from config - handle both dict and object formats
        if hasattr(self.config, 'detectors'):
            detector_configs = self.config.detectors
        else:
            detector_configs = self.config.get('detectors', [])
        
        for detector_config in detector_configs:
            # Handle both dict and object formats
            if hasattr(detector_config, 'name'):
                name = detector_config.name
                config_dict = detector_config.model_dump() if hasattr(detector_config, 'model_dump') else detector_config.__dict__
            else:
                name = detector_config.get('name')
                config_dict = detector_config
                
            try:
                if name == 'single_face':
                    detectors[name] = SingleFaceDetector(config_dict)
                elif name == 'retinaface':
                    detectors[name] = RetinaFaceDetector(config_dict)
                elif name == 'blazeface':
                    detectors[name] = BlazeFaceDetector(config_dict)
                elif name == 'opencv':
                    detectors[name] = OpenCVDetector(config_dict)
                elif name == 'mediapipe':
                    detectors[name] = MediaPipeDetector(config_dict)
            except Exception as e:
                print(f"Warning: Failed to initialize {name} detector: {e}")
                continue
        return detectors
        
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in image using fallback system."""
        # Define detector priority order (best to worst)
        detector_priority = ['mediapipe', 'opencv', 'retinaface', 'blazeface', 'single_face']
        
        # Use fallback detection strategy
        enable_fallback = self.config.get('enable_fallback', True)
        
        if enable_fallback:
            return self._detect_with_fallback(image, detector_priority)
        else:
            # Original behavior - use all detectors
            return self._detect_all_detectors(image)
    
    def _detect_with_fallback(self, image: np.ndarray, detector_priority: List[str]) -> List[Dict[str, Any]]:
        """Detect faces using fallback system - try detectors in priority order."""
        min_faces_threshold = self.config.get('min_faces_threshold', 1)
        
        for detector_name in detector_priority:
            if detector_name not in self.detectors:
                continue
                
            detector = self.detectors[detector_name]
            
            try:
                print(f"  Trying {detector_name} detector...")
                detections = detector.detect(image)
                
                if len(detections) >= min_faces_threshold:
                    print(f"  ✓ {detector_name} found {len(detections)} faces")
                    return detections
                else:
                    print(f"  → {detector_name} found only {len(detections)} faces, trying next...")
                    
            except Exception as e:
                print(f"  ✗ {detector_name} failed: {e}")
                continue
        
        # If no detector found enough faces, try manual estimation as last resort
        print("  → All detectors failed, attempting manual face estimation...")
        manual_detections = self._manual_face_estimation(image)
        
        if manual_detections:
            print(f"  ✓ Manual estimation found {len(manual_detections)} faces")
            
        return manual_detections
    
    def _detect_all_detectors(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Original detection behavior - combine all detectors."""
        all_detections = []
        
        for detector_name, detector in self.detectors.items():
            try:
                detections = detector.detect(image)
                all_detections.extend(detections)
            except Exception as e:
                print(f"Warning: {detector_name} detector failed: {e}")
                
        return all_detections
    
    def _manual_face_estimation(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Manual face estimation as last resort fallback."""
        h, w = image.shape[:2]
        
        # Estimate face location based on common portrait compositions
        # Most portraits have face in upper-center area
        
        # Estimate face size (typically 20-40% of image width)
        estimated_face_width = int(w * 0.3)
        estimated_face_height = int(estimated_face_width * 1.2)  # Face is slightly taller than wide
        
        # Estimate face position (center-upper area)
        center_x = w // 2
        face_top_y = int(h * 0.15)  # Face usually starts around 15% from top
        
        # Calculate bounding box
        x1 = center_x - estimated_face_width // 2
        y1 = face_top_y
        x2 = x1 + estimated_face_width
        y2 = y1 + estimated_face_height
        
        # Ensure bbox is within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Validate estimated bbox
        if x2 > x1 and y2 > y1 and (x2-x1) > 32 and (y2-y1) > 32:
            return [{
                'bbox': [x1, y1, x2, y2],
                'confidence': 0.5,  # Low confidence for manual estimation
                'landmarks': None,
                'detector': 'manual_estimation'
            }]
        
        return []
        
    def process_file(self, image_path: str) -> List[Dict[str, Any]]:
        """Process single image file."""
        image = cv2.imread(image_path)
        if image is None:
            return []
        return self.detect(image)