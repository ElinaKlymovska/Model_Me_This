"""
Sliding window search strategy for face detection.
"""

import numpy as np
from typing import List, Tuple, Generator, Dict, Any
import cv2


class SlidingWindowSearch:
    """Sliding window search strategy for face detection."""
    
    def __init__(self, window_size: int = 512, stride: int = 256, 
                 min_overlap: float = 0.5):
        """Initialize sliding window search.
        
        Args:
            window_size: Size of sliding window
            stride: Stride between windows
            min_overlap: Minimum overlap between windows
        """
        self.window_size = window_size
        self.stride = stride
        self.min_overlap = min_overlap
    
    def generate_windows(self, image: np.ndarray) -> Generator[Tuple[np.ndarray, Tuple[int, int]], None, None]:
        """Generate sliding windows over the image.
        
        Args:
            image: Input image
            
        Yields:
            Tuple of (window_image, (x_offset, y_offset))
        """
        height, width = image.shape[:2]
        
        # Calculate number of windows
        num_windows_h = max(1, (height - self.window_size) // self.stride + 1)
        num_windows_w = max(1, (width - self.window_size) // self.stride + 1)
        
        for i in range(num_windows_h):
            for j in range(num_windows_w):
                # Calculate window coordinates
                y1 = i * self.stride
                x1 = j * self.stride
                y2 = min(y1 + self.window_size, height)
                x2 = min(x1 + self.window_size, width)
                
                # Extract window
                window = image[y1:y2, x1:x2]
                
                # Pad if necessary to maintain window size
                if window.shape[:2] != (self.window_size, self.window_size):
                    padded_window = np.zeros((self.window_size, self.window_size, 3), dtype=window.dtype)
                    padded_window[:window.shape[0], :window.shape[1]] = window
                    window = padded_window
                
                yield window, (x1, y1)
    
    def merge_detections(self, detections: List[Dict[str, Any]], 
                        window_offsets: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Merge detections from multiple windows.
        
        Args:
            detections: List of detections from all windows
            window_offsets: List of window offsets (x, y)
            
        Returns:
            Merged detections with global coordinates
        """
        merged_detections = []
        
        for i, detection in enumerate(detections):
            if i < len(window_offsets):
                x_offset, y_offset = window_offsets[i]
                
                # Adjust bbox coordinates to global image space
                bbox = detection['bbox']
                global_bbox = [
                    bbox[0] + x_offset,
                    bbox[1] + y_offset,
                    bbox[2] + x_offset,
                    bbox[3] + y_offset
                ]
                
                # Adjust landmarks if available
                global_landmarks = None
                if detection.get('landmarks'):
                    global_landmarks = []
                    for landmark in detection['landmarks']:
                        global_landmarks.append([
                            landmark[0] + x_offset,
                            landmark[1] + y_offset
                        ])
                
                # Create merged detection
                merged_detection = detection.copy()
                merged_detection['bbox'] = global_bbox
                merged_detection['landmarks'] = global_landmarks
                merged_detection['window_offset'] = (x_offset, y_offset)
                
                merged_detections.append(merged_detection)
        
        return merged_detections
    
    def search(self, image: np.ndarray, detector) -> List[Dict[str, Any]]:
        """Perform sliding window search with face detection.
        
        Args:
            image: Input image
            detector: Face detector instance
            
        Returns:
            List of merged detections
        """
        all_detections = []
        window_offsets = []
        
        # Process each window
        for window, offset in self.generate_windows(image):
            # Detect faces in window
            window_detections = detector(window)
            
            if window_detections:
                all_detections.extend(window_detections)
                window_offsets.extend([offset] * len(window_detections))
        
        # Merge detections
        merged_detections = self.merge_detections(all_detections, window_offsets)
        
        return merged_detections
    
    def get_search_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Get information about the search strategy.
        
        Args:
            image: Input image
            
        Returns:
            Search strategy information
        """
        height, width = image.shape[:2]
        
        num_windows_h = max(1, (height - self.window_size) // self.stride + 1)
        num_windows_w = max(1, (width - self.window_size) // self.stride + 1)
        total_windows = num_windows_h * num_windows_w
        
        return {
            'strategy': 'sliding_window',
            'window_size': self.window_size,
            'stride': self.stride,
            'total_windows': total_windows,
            'image_size': (width, height),
            'coverage': self._calculate_coverage(width, height)
        }
    
    def _calculate_coverage(self, width: int, height: int) -> float:
        """Calculate coverage percentage of sliding windows.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Coverage percentage
        """
        total_area = width * height
        covered_area = 0
        
        for i in range(max(1, (height - self.window_size) // self.stride + 1)):
            for j in range(max(1, (width - self.window_size) // self.stride + 1)):
                y1 = i * self.stride
                x1 = j * self.stride
                y2 = min(y1 + self.window_size, height)
                x2 = min(x1 + self.window_size, width)
                
                covered_area += (x2 - x1) * (y2 - y1)
        
        return (covered_area / total_area) * 100
