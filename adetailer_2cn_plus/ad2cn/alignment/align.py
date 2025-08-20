"""
Face alignment utilities for detected faces.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path


class FaceAligner:
    """Face alignment utility class."""
    
    def __init__(self, desired_size: int = 256, desired_face_ratio: float = 0.75):
        """Initialize face aligner.
        
        Args:
            desired_size: Size of aligned face output
            desired_face_ratio: Ratio of face to image in aligned output
        """
        self.desired_size = desired_size
        self.desired_face_ratio = desired_face_ratio
        
        # Standard facial landmarks for alignment
        self.landmark_indices = {
            'left_eye': 0,
            'right_eye': 1,
            'nose': 2,
            'left_mouth': 3,
            'right_mouth': 4
        }
    
    def align_face(self, image: np.ndarray, landmarks: List[List[int]], 
                   bbox: List[int]) -> np.ndarray:
        """Align face using facial landmarks.
        
        Args:
            image: Input image
            landmarks: Facial landmarks as list of [x, y] coordinates
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Aligned face image
        """
        if len(landmarks) < 5:
            # Fallback to bbox-based alignment if not enough landmarks
            return self._align_by_bbox(image, bbox)
        
        # Convert landmarks to numpy array
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Calculate eye center
        left_eye = landmarks[self.landmark_indices['left_eye']]
        right_eye = landmarks[self.landmark_indices['right_eye']]
        eye_center = (left_eye + right_eye) / 2
        
        # Calculate eye angle
        eye_angle = np.degrees(np.arctan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0]
        ))
        
        # Calculate desired eye positions
        desired_left_eye_x = self.desired_size * (1 - self.desired_face_ratio) / 2
        desired_left_eye_y = self.desired_size * self.desired_face_ratio / 2
        
        # Calculate scale
        eye_distance = np.linalg.norm(right_eye - left_eye)
        desired_eye_distance = self.desired_size * self.desired_face_ratio * 0.35
        scale = desired_eye_distance / eye_distance
        
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(
            tuple(eye_center), eye_angle, scale
        )
        
        # Adjust translation
        rotation_matrix[0, 2] += desired_left_eye_x - eye_center[0]
        rotation_matrix[1, 2] += desired_left_eye_y - eye_center[1]
        
        # Apply transformation
        aligned_face = cv2.warpAffine(
            image, rotation_matrix, (self.desired_size, self.desired_size),
            flags=cv2.INTER_CUBIC
        )
        
        return aligned_face
    
    def _align_by_bbox(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Fallback alignment using bounding box.
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Cropped and resized face
        """
        x1, y1, x2, y2 = bbox
        
        # Add padding
        height, width = image.shape[:2]
        padding_x = int((x2 - x1) * 0.1)
        padding_y = int((y2 - y1) * 0.1)
        
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(width, x2 + padding_x)
        y2 = min(height, y2 + padding_y)
        
        # Crop face
        face_crop = image[y1:y2, x1:x2]
        
        # Resize to desired size
        aligned_face = cv2.resize(face_crop, (self.desired_size, self.desired_size))
        
        return aligned_face
    
    def extract_face_patches(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Extract aligned face patches from image.
        
        Args:
            image: Input image
            detections: List of face detections
            
        Returns:
            List of aligned face patches
        """
        face_patches = []
        
        for detection in detections:
            bbox = detection['bbox']
            landmarks = detection.get('landmarks')
            
            if landmarks:
                aligned_face = self.align_face(image, landmarks, bbox)
            else:
                aligned_face = self._align_by_bbox(image, bbox)
            
            face_patches.append(aligned_face)
        
        return face_patches
    
    def normalize_face(self, face_patch: np.ndarray) -> np.ndarray:
        """Normalize face patch for model input.
        
        Args:
            face_patch: Face patch image
            
        Returns:
            Normalized face patch
        """
        # Convert to float and normalize to [0, 1]
        if face_patch.dtype == np.uint8:
            face_patch = face_patch.astype(np.float32) / 255.0
        
        # Ensure proper range
        face_patch = np.clip(face_patch, 0, 1)
        
        return face_patch
