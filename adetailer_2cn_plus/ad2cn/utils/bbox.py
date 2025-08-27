"""Simple bounding box utilities."""

import numpy as np
from typing import List, Dict, Any


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    # Areas
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def apply_nms(detections: List[Dict[str, Any]], threshold: float = 0.3) -> List[Dict[str, Any]]:
    """Apply Non-Maximum Suppression."""
    if not detections:
        return []
    
    # Sort by confidence
    sorted_dets = sorted(detections, key=lambda d: d.get('confidence', 0), reverse=True)
    
    # Apply NMS
    result = []
    for det in sorted_dets:
        if not any(calculate_iou(det['bbox'], r['bbox']) > threshold for r in result):
            result.append(det)
    
    return result