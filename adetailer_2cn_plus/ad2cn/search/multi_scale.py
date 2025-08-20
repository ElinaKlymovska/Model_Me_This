"""
Multi-scale search strategy for face detection.
"""

import numpy as np
from typing import List, Tuple, Generator, Dict, Any
import cv2


class MultiScaleSearch:
    """Multi-scale search strategy for face detection."""
    
    def __init__(self, scale_factors: List[float] = None, 
                 min_size: int = 64, max_size: int = 1024):
        """Initialize multi-scale search.
        
        Args:
            scale_factors: List of scale factors to apply
            min_size: Minimum image size after scaling
            max_size: Maximum image size after scaling
        """
        if scale_factors is None:
            scale_factors = [1.0, 0.75, 0.5, 0.25]
        
        self.scale_factors = scale_factors
        self.min_size = min_size
        self.max_size = max_size
    
    def generate_scales(self, image: np.ndarray) -> Generator[Tuple[np.ndarray, float, Tuple[int, int]], None, None]:
        """Generate scaled versions of the image.
        
        Args:
            image: Input image
            
        Yields:
            Tuple of (scaled_image, scale_factor, (width, height))
        """
        height, width = image.shape[:2]
        
        for scale_factor in self.scale_factors:
            # Calculate new dimensions
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Skip if too small or too large
            if new_width < self.min_size or new_height < self.min_size:
                continue
            
            if new_width > self.max_size or new_height > self.max_size:
                continue
            
            # Resize image
            scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            yield scaled_image, scale_factor, (new_width, new_height)
    
    def merge_detections(self, detections: List[Dict[str, Any]], 
                        scale_factors: List[float]) -> List[Dict[str, Any]]:
        """Merge detections from multiple scales.
        
        Args:
            detections: List of detections from all scales
            scale_factors: List of scale factors used
            
        Returns:
            Merged detections with original scale coordinates
        """
        merged_detections = []
        
        for i, detection in enumerate(detections):
            if i < len(scale_factors):
                scale_factor = scale_factors[i]
                
                # Adjust bbox coordinates to original scale
                bbox = detection['bbox']
                original_bbox = [
                    int(bbox[0] / scale_factor),
                    int(bbox[1] / scale_factor),
                    int(bbox[2] / scale_factor),
                    int(bbox[3] / scale_factor)
                ]
                
                # Adjust landmarks if available
                original_landmarks = None
                if detection.get('landmarks'):
                    original_landmarks = []
                    for landmark in detection['landmarks']:
                        original_landmarks.append([
                            int(landmark[0] / scale_factor),
                            int(landmark[1] / scale_factor)
                        ])
                
                # Create merged detection
                merged_detection = detection.copy()
                merged_detection['bbox'] = original_bbox
                merged_detection['landmarks'] = original_landmarks
                merged_detection['scale_factor'] = scale_factor
                
                merged_detections.append(merged_detection)
        
        return merged_detections
    
    def search(self, image: np.ndarray, detector) -> List[Dict[str, Any]]:
        """Perform multi-scale search with face detection.
        
        Args:
            image: Input image
            detector: Face detector instance
            
        Returns:
            List of merged detections
        """
        all_detections = []
        scale_factors = []
        
        # Process each scale
        for scaled_image, scale_factor, (width, height) in self.generate_scales(image):
            # Detect faces in scaled image
            scale_detections = detector(scaled_image)
            
            if scale_detections:
                all_detections.extend(scale_detections)
                scale_factors.extend([scale_factor] * len(scale_detections))
        
        # Merge detections
        merged_detections = self.merge_detections(all_detections, scale_factors)
        
        return merged_detections
    
    def get_search_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Get information about the search strategy.
        
        Args:
            image: Input image
            
        Returns:
            Search strategy information
        """
        height, width = image.shape[:2]
        
        # Calculate effective scales
        effective_scales = []
        for scale_factor in self.scale_factors:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            if (self.min_size <= new_width <= self.max_size and 
                self.min_size <= new_height <= self.max_size):
                effective_scales.append({
                    'scale_factor': scale_factor,
                    'size': (new_width, new_height)
                })
        
        return {
            'strategy': 'multi_scale',
            'scale_factors': self.scale_factors,
            'effective_scales': effective_scales,
            'total_scales': len(effective_scales),
            'original_size': (width, height),
            'min_size': self.min_size,
            'max_size': self.max_size
        }
    
    def optimize_scales(self, image: np.ndarray, target_faces: int = 10) -> List[float]:
        """Optimize scale factors based on image content and target face count.
        
        Args:
            image: Input image
            target_faces: Target number of faces to detect
            
        Returns:
            Optimized list of scale factors
        """
        height, width = image.shape[:2]
        
        # Calculate optimal scales based on image size
        optimal_scales = []
        
        # Start with original scale
        optimal_scales.append(1.0)
        
        # Add scales for smaller faces
        if width > 1024 or height > 1024:
            optimal_scales.extend([0.75, 0.5])
        
        if width > 2048 or height > 2048:
            optimal_scales.extend([0.25, 0.125])
        
        # Add scales for larger faces
        if width < 512 or height < 512:
            optimal_scales.extend([1.5, 2.0])
        
        # Ensure we have reasonable number of scales
        if len(optimal_scales) < 3:
            optimal_scales.extend([0.75, 0.5])
        
        # Sort scales from largest to smallest
        optimal_scales.sort(reverse=True)
        
        return optimal_scales[:6]  # Limit to 6 scales max
