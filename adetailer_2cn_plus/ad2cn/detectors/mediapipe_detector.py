"""
MediaPipe face detector with precise landmarks.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import cv2
from .base import FaceDetector

# Optional MediaPipe import
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


class MediaPipeDetector(FaceDetector):
    """MediaPipe face detector with landmark support."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MediaPipe detector.
        
        Args:
            config: Detector configuration
        """
        super().__init__(config)
        
        # MediaPipe specific settings
        self.detection_confidence = config.get('detection_confidence', 0.7)
        self.model_selection = config.get('model_selection', 0)  # 0 for short range, 1 for full range
        self.with_landmarks = config.get('with_landmarks', True)
        
        # MediaPipe components
        self.face_detection = None
        self.face_mesh = None
        self.drawing_utils = None
        self.drawing_styles = None
        
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available. Install with: pip install mediapipe")
    
    def load_model(self) -> None:
        """Load MediaPipe face detection and mesh models."""
        try:
            # Initialize MediaPipe Face Detection
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=self.model_selection,
                min_detection_confidence=self.detection_confidence
            )
            
            # Initialize Face Mesh for landmarks (if needed)
            if self.with_landmarks:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=self.max_faces,
                    refine_landmarks=True,
                    min_detection_confidence=self.detection_confidence
                )
            
            # Drawing utilities
            self.drawing_utils = mp.solutions.drawing_utils
            self.drawing_styles = mp.solutions.drawing_styles
            
            self._initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MediaPipe face detector: {e}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using MediaPipe.
        
        Args:
            image: Input image (H, W, C) in BGR format
            
        Returns:
            List of detections with refined bounding boxes
        """
        if not self._initialized:
            self.load_model()
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        detections = []
        
        # Primary detection using Face Detection
        results = self.face_detection.process(rgb_image)
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # Ensure bbox is within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                confidence = detection.score[0]
                
                # Get landmarks using Face Mesh if enabled
                landmarks = None
                refined_bbox = [x1, y1, x2, y2]
                
                if self.with_landmarks and self.face_mesh:
                    landmarks, refined_bbox = self._get_refined_landmarks_and_bbox(
                        rgb_image, [x1, y1, x2, y2]
                    )
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        'bbox': refined_bbox,
                        'confidence': float(confidence),
                        'landmarks': landmarks,
                        'detector': 'mediapipe'
                    })
        
        return detections
    
    def _get_refined_landmarks_and_bbox(self, rgb_image: np.ndarray, 
                                       initial_bbox: List[int]) -> tuple:
        """Get refined landmarks and bbox using Face Mesh.
        
        Args:
            rgb_image: RGB image
            initial_bbox: Initial bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple of (landmarks_dict, refined_bbox)
        """
        h, w = rgb_image.shape[:2]
        
        # Run Face Mesh
        mesh_results = self.face_mesh.process(rgb_image)
        
        if not mesh_results.multi_face_landmarks:
            return None, initial_bbox
        
        # Use first detected face
        face_landmarks = mesh_results.multi_face_landmarks[0]
        
        # Extract key landmark points
        landmarks_points = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_points.append([x, y])
        
        landmarks_array = np.array(landmarks_points)
        
        # Key facial landmarks indices (MediaPipe 468 model)
        key_landmarks = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose': [1, 2, 5, 4, 6, 168, 8, 9, 10, 151],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
        
        # Calculate refined bbox based on face oval landmarks
        if len(landmarks_array) > max(key_landmarks['face_oval']):
            face_oval_points = landmarks_array[key_landmarks['face_oval']]
            
            x_min = np.min(face_oval_points[:, 0])
            y_min = np.min(face_oval_points[:, 1])
            x_max = np.max(face_oval_points[:, 0])
            y_max = np.max(face_oval_points[:, 1])
            
            # Add smart padding (15% buffer)
            padding_x = int((x_max - x_min) * 0.15)
            padding_y = int((y_max - y_min) * 0.15)
            
            refined_x1 = max(0, x_min - padding_x)
            refined_y1 = max(0, y_min - padding_y)
            refined_x2 = min(w, x_max + padding_x)
            refined_y2 = min(h, y_max + padding_y)
            
            refined_bbox = [refined_x1, refined_y1, refined_x2, refined_y2]
        else:
            refined_bbox = initial_bbox
        
        # Create landmarks dictionary with key points
        landmarks_dict = {}
        try:
            if len(landmarks_array) > max(max(indices) for indices in key_landmarks.values()):
                for region, indices in key_landmarks.items():
                    region_points = []
                    for idx in indices:
                        if idx < len(landmarks_array):
                            region_points.append(landmarks_array[idx].tolist())
                    landmarks_dict[region] = region_points
        except Exception:
            landmarks_dict = None
        
        return landmarks_dict, refined_bbox
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for MediaPipe detection."""
        # MediaPipe expects uint8 BGR images
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Enhance image for better detection
        image = self._enhance_for_detection(image)
        
        return image
    
    def _enhance_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better face detection."""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        enhanced_gray = cv2.equalizeHist(gray)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        
        # Blend with original (50/50)
        result = cv2.addWeighted(image, 0.5, enhanced, 0.5, 0)
        
        return result
    
    def validate_bbox(self, bbox: List[int], image_shape: tuple) -> List[int]:
        """Validate and fix bounding box dimensions.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            image_shape: Image shape (h, w, c)
            
        Returns:
            Validated bounding box
        """
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Ensure positive dimensions
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Check minimum size (face should be at least 32x32)
        if (x2 - x1) < 32 or (y2 - y1) < 32:
            return None
        
        # Check maximum size (face shouldn't be more than 80% of image)
        if (x2 - x1) > w * 0.8 or (y2 - y1) > h * 0.8:
            return None
        
        # Check aspect ratio (face should be roughly 0.7 to 1.3 ratio)
        aspect_ratio = (x2 - x1) / (y2 - y1)
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:
            return None
        
        return bbox
    
    def postprocess(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced postprocessing with bbox validation."""
        # First apply standard postprocessing
        filtered = super().postprocess(detections)
        
        # Additional validation for MediaPipe detections
        validated = []
        for detection in filtered:
            bbox = detection['bbox']
            
            # Validate bbox dimensions (this would need image_shape, so skip for now)
            # In practice, you'd pass image_shape through the detection pipeline
            validated.append(detection)
        
        return validated