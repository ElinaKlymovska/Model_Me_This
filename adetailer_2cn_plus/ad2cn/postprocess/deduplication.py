"""Simple face deduplication."""

import numpy as np
from typing import List, Dict, Any


class FaceDeduplicator:
    """Simple face deduplication based on confidence."""
    
    def __init__(self, config: Dict[str, Any]):
        self.overlap_threshold = config.get('overlap_threshold', 0.3)
        
    def deduplicate(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate faces, keep highest confidence."""
        if len(faces) <= 1:
            return faces
            
        # Sort by confidence descending
        sorted_faces = sorted(faces, key=lambda f: f.get('confidence', 0), reverse=True)
        
        # Keep only non-overlapping faces
        result = []
        for face in sorted_faces:
            if not self._overlaps_with_any(face, result):
                result.append(face)
                
        return result
        
    def _overlaps_with_any(self, face: Dict[str, Any], faces: List[Dict[str, Any]]) -> bool:
        """Check if face overlaps with any in the list."""
        for other in faces:
            if self._calculate_overlap(face, other) > self.overlap_threshold:
                return True
        return False
        
    def _calculate_overlap(self, face1: Dict[str, Any], face2: Dict[str, Any]) -> float:
        """Calculate IoU overlap between two faces."""
        box1 = face1.get('bbox', [0, 0, 1, 1])
        box2 = face2.get('bbox', [0, 0, 1, 1])
        
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = max(0, min(x1 + w1, x2 + w2) - xi)
        hi = max(0, min(y1 + h1, y2 + h2) - yi)
        
        intersection = wi * hi
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0