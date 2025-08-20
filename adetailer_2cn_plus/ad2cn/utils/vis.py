"""
Visualization utilities for debug overlays.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure


def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]], 
                   show_confidence: bool = True, show_landmarks: bool = True,
                   colors: Optional[Dict[str, Tuple[int, int, int]]] = None) -> np.ndarray:
    """Draw face detections on image.
    
    Args:
        image: Input image
        detections: List of face detections
        show_confidence: Whether to show confidence scores
        show_landmarks: Whether to show facial landmarks
        colors: Dictionary mapping detector names to BGR colors
        
    Returns:
        Image with detection overlays
    """
    if colors is None:
        colors = {
            'blazeface': (0, 255, 0),      # Green
            'retinaface': (255, 0, 0),     # Blue
            'mtcnn': (0, 0, 255),          # Red
            'scrfd': (255, 255, 0),        # Cyan
            'default': (0, 255, 255)       # Yellow
        }
    
    # Create a copy to avoid modifying original
    result = image.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection.get('confidence', 0.0)
        detector = detection.get('detector', 'default')
        landmarks = detection.get('landmarks')
        
        # Get color for this detector
        color = colors.get(detector, colors['default'])
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw detector label
        label = f"{detector}"
        if show_confidence:
            label += f" {confidence:.2f}"
        
        # Calculate text position
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1
        text_y = y1 - 10 if y1 > 20 else y1 + text_size[1] + 10
        
        # Draw text background
        cv2.rectangle(result, 
                     (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0], text_y + 5),
                     color, -1)
        
        # Draw text
        cv2.putText(result, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw landmarks if available
        if show_landmarks and landmarks:
            for i, landmark in enumerate(landmarks):
                x, y = map(int, landmark)
                cv2.circle(result, (x, y), 3, (255, 255, 255), -1)
                cv2.circle(result, (x, y), 3, color, 1)
    
    return result


def draw_search_windows(image: np.ndarray, search_info: Dict[str, Any],
                       color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """Draw search strategy information on image.
    
    Args:
        image: Input image
        search_info: Search strategy information
        color: Color for window overlays
        
    Returns:
        Image with search window overlays
    """
    result = image.copy()
    
    if search_info['strategy'] == 'sliding_window':
        result = _draw_sliding_windows(result, search_info, color)
    elif search_info['strategy'] == 'multi_scale':
        result = _draw_multi_scale_info(result, search_info, color)
    
    return result


def _draw_sliding_windows(image: np.ndarray, search_info: Dict[str, Any],
                         color: Tuple[int, int, int]) -> np.ndarray:
    """Draw sliding window overlays.
    
    Args:
        image: Input image
        search_info: Search strategy information
        color: Color for window overlays
        
    Returns:
        Image with window overlays
    """
    result = image.copy()
    height, width = image.shape[:2]
    
    window_size = search_info['window_size']
    stride = search_info['stride']
    
    # Draw window boundaries
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            x1, y1 = j, i
            x2 = min(j + window_size, width)
            y2 = min(i + window_size, height)
            
            # Draw window rectangle
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 1)
    
    # Add search info text
    info_text = f"Windows: {search_info['total_windows']}, Coverage: {search_info['coverage']:.1f}%"
    cv2.putText(result, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return result


def _draw_multi_scale_info(image: np.ndarray, search_info: Dict[str, Any],
                          color: Tuple[int, int, int]) -> np.ndarray:
    """Draw multi-scale search information.
    
    Args:
        image: Input image
        search_info: Search strategy information
        color: Color for overlays
        
    Returns:
        Image with scale information
    """
    result = image.copy()
    
    # Add scale information text
    y_offset = 30
    cv2.putText(result, f"Multi-scale Search", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    y_offset += 30
    for scale_info in search_info['effective_scales']:
        scale_text = f"Scale {scale_info['scale_factor']}: {scale_info['size']}"
        cv2.putText(result, scale_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20
    
    return result


def create_detection_summary(detections: List[Dict[str, Any]], 
                           image_path: Optional[str] = None) -> Figure:
    """Create a summary visualization of detections.
    
    Args:
        detections: List of face detections
        image_path: Optional image path for context
        
    Returns:
        Matplotlib figure with detection summary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Detection count by detector
    detector_counts = {}
    confidence_scores = []
    
    for detection in detections:
        detector = detection.get('detector', 'unknown')
        detector_counts[detector] = detector_counts.get(detector, 0) + 1
        confidence_scores.append(detection.get('confidence', 0.0))
    
    # Plot detector counts
    if detector_counts:
        ax1.bar(detector_counts.keys(), detector_counts.values())
        ax1.set_title('Face Detections by Detector')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot confidence distribution
    if confidence_scores:
        ax2.hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Confidence Score Distribution')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(confidence_scores):.3f}')
        ax2.legend()
    
    plt.tight_layout()
    return fig


def save_visualization(image: np.ndarray, output_path: str, 
                      detections: Optional[List[Dict[str, Any]]] = None,
                      search_info: Optional[Dict[str, Any]] = None) -> None:
    """Save visualization with optional overlays.
    
    Args:
        image: Input image
        output_path: Output file path
        detections: Optional face detections to overlay
        search_info: Optional search strategy information to overlay
    """
    result = image.copy()
    
    # Add detection overlays
    if detections:
        result = draw_detections(result, detections)
    
    # Add search strategy overlays
    if search_info:
        result = draw_search_windows(result, search_info)
    
    # Save result
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def create_comparison_grid(images: List[np.ndarray], 
                          titles: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> Figure:
    """Create a grid comparison of multiple images.
    
    Args:
        images: List of images to compare
        titles: Optional list of titles for each image
        figsize: Figure size
        
    Returns:
        Matplotlib figure with image grid
    """
    n_images = len(images)
    
    # Calculate grid dimensions
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single row/column case
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (image, ax) in enumerate(zip(images, axes.flat)):
        ax.imshow(image)
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_images, rows * cols):
        axes.flat[i].axis('off')
    
    plt.tight_layout()
    return fig
