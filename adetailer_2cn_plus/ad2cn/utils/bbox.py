"""
Bounding box utilities and operations.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import cv2


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there is intersection
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou


def apply_nms(detections: List[Dict[str, Any]], 
              iou_threshold: float = 0.3,
              confidence_threshold: float = 0.0) -> List[Dict[str, Any]]:
    """Apply Non-Maximum Suppression (NMS) to face detections.
    
    Args:
        detections: List of face detections
        iou_threshold: IoU threshold for suppression
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Filtered list of detections after NMS
    """
    if not detections:
        return []
    
    # Filter by confidence threshold
    filtered_detections = [
        det for det in detections 
        if det.get('confidence', 0.0) >= confidence_threshold
    ]
    
    if not filtered_detections:
        return []
    
    # Sort by confidence (descending)
    filtered_detections.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
    
    # Apply NMS
    kept_detections = []
    
    while filtered_detections:
        # Keep the detection with highest confidence
        current = filtered_detections.pop(0)
        kept_detections.append(current)
        
        # Remove overlapping detections
        filtered_detections = [
            det for det in filtered_detections
            if calculate_iou(current['bbox'], det['bbox']) <= iou_threshold
        ]
    
    return kept_detections


def merge_overlapping_boxes(detections: List[Dict[str, Any]], 
                           iou_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """Merge overlapping bounding boxes.
    
    Args:
        detections: List of face detections
        iou_threshold: IoU threshold for merging
        
    Returns:
        List of detections with merged boxes
    """
    if not detections:
        return []
    
    # Sort by confidence (descending)
    sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0.0), reverse=True)
    
    merged_detections = []
    used_indices = set()
    
    for i, detection in enumerate(sorted_detections):
        if i in used_indices:
            continue
        
        current_box = detection['bbox']
        current_confidence = detection.get('confidence', 0.0)
        merged_indices = [i]
        
        # Find overlapping detections
        for j, other_detection in enumerate(sorted_detections[i+1:], i+1):
            if j in used_indices:
                continue
            
            other_box = other_detection['bbox']
            iou = calculate_iou(current_box, other_box)
            
            if iou >= iou_threshold:
                merged_indices.append(j)
                used_indices.add(j)
        
        # Merge overlapping boxes
        if len(merged_indices) > 1:
            merged_box = _merge_boxes([sorted_detections[idx]['bbox'] for idx in merged_indices])
            merged_confidence = max(sorted_detections[idx].get('confidence', 0.0) 
                                  for idx in merged_indices)
            
            # Create merged detection
            merged_detection = detection.copy()
            merged_detection['bbox'] = merged_box
            merged_detection['confidence'] = merged_confidence
            merged_detection['merged_count'] = len(merged_indices)
            
            merged_detections.append(merged_detection)
        else:
            merged_detections.append(detection)
        
        used_indices.add(i)
    
    return merged_detections


def _merge_boxes(boxes: List[List[float]]) -> List[float]:
    """Merge multiple bounding boxes into one.
    
    Args:
        boxes: List of bounding boxes [x1, y1, x2, y2]
        
    Returns:
        Merged bounding box [x1, y1, x2, y2]
    """
    if not boxes:
        return [0, 0, 0, 0]
    
    # Calculate union of all boxes
    x1_min = min(box[0] for box in boxes)
    y1_min = min(box[1] for box in boxes)
    x2_max = max(box[2] for box in boxes)
    y2_max = max(box[3] for box in boxes)
    
    return [x1_min, y1_min, x2_max, y2_max]


def expand_bbox(bbox: List[float], expansion_factor: float = 0.1,
                image_shape: Optional[Tuple[int, int]] = None) -> List[float]:
    """Expand bounding box by a factor.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        expansion_factor: Factor to expand the box (0.1 = 10% expansion)
        image_shape: Optional image shape (height, width) for boundary checking
        
    Returns:
        Expanded bounding box [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate expansion amounts
    width = x2 - x1
    height = y2 - y1
    
    expand_x = int(width * expansion_factor)
    expand_y = int(height * expansion_factor)
    
    # Apply expansion
    new_x1 = max(0, x1 - expand_x)
    new_y1 = max(0, y1 - expand_y)
    new_x2 = x2 + expand_x
    new_y2 = y2 + expand_y
    
    # Clip to image boundaries if provided
    if image_shape:
        height, width = image_shape
        new_x1 = max(0, min(new_x1, width))
        new_y1 = max(0, min(new_y1, height))
        new_x2 = max(0, min(new_x2, width))
        new_y2 = max(0, min(new_y2, height))
    
    return [new_x1, new_y1, new_x2, new_y2]


def calculate_bbox_area(bbox: List[float]) -> float:
    """Calculate area of bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Area of the bounding box
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def calculate_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Calculate center point of bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Center point (x, y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def filter_boxes_by_size(detections: List[Dict[str, Any]], 
                         min_area: float = 0.0,
                         max_area: float = float('inf')) -> List[Dict[str, Any]]:
    """Filter detections by bounding box area.
    
    Args:
        detections: List of face detections
        min_area: Minimum area threshold
        max_area: Maximum area threshold
        
    Returns:
        Filtered list of detections
    """
    filtered_detections = []
    
    for detection in detections:
        bbox = detection['bbox']
        area = calculate_bbox_area(bbox)
        
        if min_area <= area <= max_area:
            filtered_detections.append(detection)
    
    return filtered_detections


def sort_detections_by_area(detections: List[Dict[str, Any]], 
                           reverse: bool = False) -> List[Dict[str, Any]]:
    """Sort detections by bounding box area.
    
    Args:
        detections: List of face detections
        reverse: If True, sort in descending order
        
    Returns:
        Sorted list of detections
    """
    return sorted(detections, 
                 key=lambda x: calculate_bbox_area(x['bbox']), 
                 reverse=reverse)
