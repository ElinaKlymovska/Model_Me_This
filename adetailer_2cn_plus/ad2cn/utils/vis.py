"""Simple visualization utilities for face detection."""
import cv2
import numpy as np
from typing import List, Dict, Any

def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes on image."""
    result = image.copy()
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection.get('confidence', 0.0)
        
        # Draw rectangle
        cv2.rectangle(result, (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        # Draw confidence text
        cv2.putText(result, f'{confidence:.3f}', 
                   (int(bbox[0]), int(bbox[1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return result

def save_visualization(image: np.ndarray, output_path: str, detections: List[Dict[str, Any]]):
    """Save image with detection visualizations."""
    viz_image = draw_detections(image, detections)
    cv2.imwrite(output_path, viz_image)