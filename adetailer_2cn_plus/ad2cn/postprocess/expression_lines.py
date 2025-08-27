"""Expression lines and facial aging system."""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum


class ExpressionType(Enum):
    """Types of facial expressions."""
    NEUTRAL = "neutral"
    SMILING = "smiling"


class ExpressionProcessor:
    """System for adding realistic expression lines and facial aging."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.neutral_intensity = config.get('expression_lines', {}).get('neutral_intensity', 0.3)
        self.smile_intensity = config.get('expression_lines', {}).get('smile_intensity', 0.6)
        
        # Person-specific expression settings
        self.person_profiles = config.get('expression_lines', {}).get('person_profiles', {
            'massy': {'expression_strength': 0.5},
            'orbi': {'expression_strength': 0.6}, 
            'yana': {'expression_strength': 0.55}
        })
        
    def _get_person_profile(self, image_path: Optional[str] = None) -> Dict[str, float]:
        """Get person-specific expression settings."""
        if image_path:
            filename = image_path.lower()
            for person, profile in self.person_profiles.items():
                if person in filename:
                    return profile
        
        return {'expression_strength': 0.5}
    
    def apply_expression_lines(self, image: np.ndarray, face_bbox: List[int],
                             expression: ExpressionType = ExpressionType.NEUTRAL,
                             image_path: Optional[str] = None) -> np.ndarray:
        """Apply expression lines based on expression type."""
        x1, y1, x2, y2 = face_bbox
        face_region = image[y1:y2, x1:x2].copy()
        
        if face_region.size == 0:
            return image
            
        profile = self._get_person_profile(image_path)
        
        if expression == ExpressionType.NEUTRAL:
            enhanced_face = self._apply_neutral_lines(face_region, profile)
        else:  # SMILING
            enhanced_face = self._apply_smiling_lines(face_region, profile)
        
        # Blend back into original image
        result = image.copy()
        blend_mask = self._create_soft_blend_mask(face_region.shape[:2])
        
        for c in range(3):
            face_channel = enhanced_face[:, :, c].astype(np.float32)
            orig_channel = result[y1:y2, x1:x2, c].astype(np.float32)
            
            blended = blend_mask * face_channel + (1 - blend_mask) * orig_channel
            result[y1:y2, x1:x2, c] = np.clip(blended, 0, 255).astype(np.uint8)
        
        return result
    
    def _apply_neutral_lines(self, face: np.ndarray, profile: Dict[str, float]) -> np.ndarray:
        """Apply neutral expression lines (subtle aging)."""
        h, w = face.shape[:2]
        result = face.copy()
        intensity = self.neutral_intensity * profile['expression_strength']
        
        # 1. Subtle horizontal forehead lines
        result = self._add_forehead_lines(result, intensity * 0.8, subtle=True)
        
        # 2. Small crow's feet (barely noticeable)
        result = self._add_crows_feet(result, intensity * 0.6, subtle=True)
        
        # 3. Vertical lines between eyebrows (weak)
        result = self._add_frown_lines(result, intensity * 0.5)
        
        # 4. Light nasolabial folds
        result = self._add_nasolabial_folds(result, intensity * 0.7, subtle=True)
        
        # 5. Small vertical chin fold
        result = self._add_chin_fold(result, intensity * 0.4)
        
        return result
    
    def _apply_smiling_lines(self, face: np.ndarray, profile: Dict[str, float]) -> np.ndarray:
        """Apply smiling expression lines (more pronounced)."""
        h, w = face.shape[:2]
        result = face.copy()
        intensity = self.smile_intensity * profile['expression_strength']
        
        # 1. Clear crow's feet from outer eye corners
        result = self._add_crows_feet(result, intensity * 0.9, subtle=False)
        
        # 2. Horizontal folds under lower eyelids
        result = self._add_under_eye_folds(result, intensity * 0.7)
        
        # 3. Pronounced nasolabial folds
        result = self._add_nasolabial_folds(result, intensity * 0.85, subtle=False)
        
        # 4. Enhanced perioral wrinkles around mouth
        result = self._add_perioral_wrinkles(result, intensity * 0.7)
        
        # 5. Radial lines in mouth corners  
        result = self._add_smile_lines(result, intensity * 0.8)
        
        # 6. Upper lip vertical lines
        result = self._add_upper_lip_lines(result, intensity * 0.6)
        
        # 7. Cheek elevation folds under eyes
        result = self._add_cheek_elevation_folds(result, intensity * 0.6)
        
        # 8. Chin and neck tension
        result = self._add_chin_tension(result, intensity * 0.5)
        
        return result
    
    def _add_forehead_lines(self, face: np.ndarray, intensity: float, subtle: bool = True) -> np.ndarray:
        """Add horizontal forehead lines."""
        h, w = face.shape[:2]
        result = face.copy().astype(np.float32)
        
        # Create forehead line mask
        line_mask = np.zeros((h, w), dtype=np.float32)
        
        line_positions = [0.12, 0.16, 0.20] if not subtle else [0.14, 0.18]
        line_thickness = 1 if subtle else 2
        
        for pos in line_positions:
            y = int(h * pos)
            # Create slightly curved horizontal line
            for x in range(int(w*0.15), int(w*0.85)):
                curve_offset = int(2 * np.sin((x - w*0.15) / (w*0.7) * np.pi))
                actual_y = y + curve_offset
                if 0 <= actual_y < h:
                    line_mask[actual_y-line_thickness:actual_y+line_thickness+1, x] = intensity
        
        # Apply darkening effect
        result = self._apply_line_effect(result, line_mask)
        return result
    
    def _add_crows_feet(self, face: np.ndarray, intensity: float, subtle: bool = True) -> np.ndarray:
        """Add crow's feet around eyes."""
        h, w = face.shape[:2]
        result = face.copy().astype(np.float32)
        
        line_mask = np.zeros((h, w), dtype=np.float32)
        
        # Left eye crow's feet
        left_eye_x, left_eye_y = int(w * 0.25), int(h * 0.38)
        num_lines = 2 if subtle else 3
        
        for i in range(num_lines):
            angle = -30 + i * 15  # Fan pattern
            length = int(w * (0.04 if subtle else 0.06))
            end_x = left_eye_x + int(length * np.cos(np.radians(angle)))
            end_y = left_eye_y + int(length * np.sin(np.radians(angle)))
            
            cv2.line(line_mask, (left_eye_x, left_eye_y), (end_x, end_y), 
                    intensity * (0.8 - i * 0.1), 1)
        
        # Right eye crow's feet
        right_eye_x, right_eye_y = int(w * 0.75), int(h * 0.38)
        
        for i in range(num_lines):
            angle = 210 - i * 15  # Fan pattern (mirrored)
            length = int(w * (0.04 if subtle else 0.06))
            end_x = right_eye_x + int(length * np.cos(np.radians(angle)))
            end_y = right_eye_y + int(length * np.sin(np.radians(angle)))
            
            cv2.line(line_mask, (right_eye_x, right_eye_y), (end_x, end_y),
                    intensity * (0.8 - i * 0.1), 1)
        
        result = self._apply_line_effect(result, line_mask)
        return result
    
    def _add_frown_lines(self, face: np.ndarray, intensity: float) -> np.ndarray:
        """Add vertical frown lines between eyebrows."""
        h, w = face.shape[:2]
        result = face.copy().astype(np.float32)
        
        line_mask = np.zeros((h, w), dtype=np.float32)
        
        # Central frown line
        center_x = int(w * 0.5)
        start_y = int(h * 0.25)
        end_y = int(h * 0.35)
        
        cv2.line(line_mask, (center_x, start_y), (center_x, end_y), intensity, 1)
        
        # Side frown lines (lighter)
        left_x = int(w * 0.47)
        right_x = int(w * 0.53)
        
        cv2.line(line_mask, (left_x, start_y+2), (left_x, end_y-2), intensity * 0.6, 1)
        cv2.line(line_mask, (right_x, start_y+2), (right_x, end_y-2), intensity * 0.6, 1)
        
        result = self._apply_line_effect(result, line_mask)
        return result
    
    def _add_nasolabial_folds(self, face: np.ndarray, intensity: float, subtle: bool = True) -> np.ndarray:
        """Add nasolabial folds from nose to mouth corners."""
        h, w = face.shape[:2]
        result = face.copy().astype(np.float32)
        
        line_mask = np.zeros((h, w), dtype=np.float32)
        thickness = 1 if subtle else 2
        
        # Left nasolabial fold
        left_nose_x, left_nose_y = int(w * 0.45), int(h * 0.55)
        left_mouth_x, left_mouth_y = int(w * 0.35), int(h * 0.75)
        
        # Create curved line using multiple points
        points = []
        for t in np.linspace(0, 1, 20):
            # Bezier-like curve
            control_x = int(w * 0.38)
            control_y = int(h * 0.65)
            
            x = int((1-t)**2 * left_nose_x + 2*(1-t)*t * control_x + t**2 * left_mouth_x)
            y = int((1-t)**2 * left_nose_y + 2*(1-t)*t * control_y + t**2 * left_mouth_y)
            points.append((x, y))
        
        for i in range(len(points)-1):
            cv2.line(line_mask, points[i], points[i+1], intensity, thickness)
        
        # Right nasolabial fold
        right_nose_x, right_nose_y = int(w * 0.55), int(h * 0.55)
        right_mouth_x, right_mouth_y = int(w * 0.65), int(h * 0.75)
        
        points = []
        for t in np.linspace(0, 1, 20):
            control_x = int(w * 0.62)
            control_y = int(h * 0.65)
            
            x = int((1-t)**2 * right_nose_x + 2*(1-t)*t * control_x + t**2 * right_mouth_x)
            y = int((1-t)**2 * right_nose_y + 2*(1-t)*t * control_y + t**2 * right_mouth_y)
            points.append((x, y))
        
        for i in range(len(points)-1):
            cv2.line(line_mask, points[i], points[i+1], intensity, thickness)
        
        result = self._apply_line_effect(result, line_mask)
        return result
    
    def _add_under_eye_folds(self, face: np.ndarray, intensity: float) -> np.ndarray:
        """Add horizontal folds under lower eyelids."""
        h, w = face.shape[:2]
        result = face.copy().astype(np.float32)
        
        line_mask = np.zeros((h, w), dtype=np.float32)
        
        # Left under-eye fold
        left_start_x, left_y = int(w * 0.32), int(h * 0.45)
        left_end_x = int(w * 0.42)
        
        cv2.line(line_mask, (left_start_x, left_y), (left_end_x, left_y), intensity, 1)
        
        # Right under-eye fold
        right_start_x, right_y = int(w * 0.58), int(h * 0.45)
        right_end_x = int(w * 0.68)
        
        cv2.line(line_mask, (right_start_x, right_y), (right_end_x, right_y), intensity, 1)
        
        result = self._apply_line_effect(result, line_mask)
        return result
    
    def _add_smile_lines(self, face: np.ndarray, intensity: float) -> np.ndarray:
        """Add radial smile lines around mouth corners."""
        h, w = face.shape[:2]
        result = face.copy().astype(np.float32)
        
        line_mask = np.zeros((h, w), dtype=np.float32)
        
        # Left mouth corner lines
        left_corner_x, left_corner_y = int(w * 0.35), int(h * 0.75)
        
        for i, angle in enumerate([-45, -30, -15]):
            length = int(w * 0.025)
            end_x = left_corner_x + int(length * np.cos(np.radians(angle)))
            end_y = left_corner_y + int(length * np.sin(np.radians(angle)))
            
            cv2.line(line_mask, (left_corner_x, left_corner_y), (end_x, end_y),
                    intensity * (0.8 - i * 0.1), 1)
        
        # Right mouth corner lines
        right_corner_x, right_corner_y = int(w * 0.65), int(h * 0.75)
        
        for i, angle in enumerate([225, 210, 195]):
            length = int(w * 0.025)
            end_x = right_corner_x + int(length * np.cos(np.radians(angle)))
            end_y = right_corner_y + int(length * np.sin(np.radians(angle)))
            
            cv2.line(line_mask, (right_corner_x, right_corner_y), (end_x, end_y),
                    intensity * (0.8 - i * 0.1), 1)
        
        result = self._apply_line_effect(result, line_mask)
        return result
    
    def _add_perioral_wrinkles(self, face: np.ndarray, intensity: float) -> np.ndarray:
        """Add comprehensive perioral wrinkles around the mouth area."""
        h, w = face.shape[:2]
        result = face.copy().astype(np.float32)
        
        line_mask = np.zeros((h, w), dtype=np.float32)
        
        # Enhanced mouth corner wrinkles (more detailed than basic smile lines)
        left_corner_x, left_corner_y = int(w * 0.35), int(h * 0.75)
        right_corner_x, right_corner_y = int(w * 0.65), int(h * 0.75)
        
        # Left corner - multiple radial lines with varying lengths and intensities
        corner_angles_left = [-60, -45, -30, -15, 0, 15]
        for i, angle in enumerate(corner_angles_left):
            # Varying line lengths for natural look
            base_length = int(w * 0.02)
            length = base_length + int(base_length * 0.3 * np.sin(i * np.pi / len(corner_angles_left)))
            
            end_x = left_corner_x + int(length * np.cos(np.radians(angle)))
            end_y = left_corner_y + int(length * np.sin(np.radians(angle)))
            
            # Varying intensity - central lines stronger
            line_intensity = intensity * (0.9 - abs(i - len(corner_angles_left)//2) * 0.1)
            cv2.line(line_mask, (left_corner_x, left_corner_y), (end_x, end_y), line_intensity, 1)
        
        # Right corner - mirrored pattern
        corner_angles_right = [180-60, 180-45, 180-30, 180-15, 180, 180+15]
        for i, angle in enumerate(corner_angles_right):
            base_length = int(w * 0.02)
            length = base_length + int(base_length * 0.3 * np.sin(i * np.pi / len(corner_angles_right)))
            
            end_x = right_corner_x + int(length * np.cos(np.radians(angle)))
            end_y = right_corner_y + int(length * np.sin(np.radians(angle)))
            
            line_intensity = intensity * (0.9 - abs(i - len(corner_angles_right)//2) * 0.1)
            cv2.line(line_mask, (right_corner_x, right_corner_y), (end_x, end_y), line_intensity, 1)
        
        # Lower lip wrinkles (horizontal lines under lower lip)
        lower_lip_center_x, lower_lip_y = int(w * 0.5), int(h * 0.78)
        
        for i in range(3):  # 3 horizontal lines of varying length
            y_offset = i * 2 + 2
            line_y = lower_lip_y + y_offset
            
            # Central line longer, side lines shorter
            if i == 0:  # Central line
                start_x, end_x = int(w * 0.45), int(w * 0.55)
                line_intensity = intensity * 0.6
            else:  # Side lines
                start_x, end_x = int(w * 0.47), int(w * 0.53)
                line_intensity = intensity * (0.5 - i * 0.1)
            
            cv2.line(line_mask, (start_x, line_y), (end_x, line_y), line_intensity, 1)
        
        # Marionette lines (lines from mouth corners down towards chin)
        # Left marionette line
        marionette_end_left_x = left_corner_x - int(w * 0.02)
        marionette_end_left_y = left_corner_y + int(h * 0.08)
        
        # Create curved marionette line
        points_left = []
        for t in np.linspace(0, 1, 15):
            # Slight curve outward then inward
            curve_offset = int(w * 0.01 * np.sin(t * np.pi))
            x = int((1-t) * left_corner_x + t * marionette_end_left_x) - curve_offset
            y = int((1-t) * left_corner_y + t * marionette_end_left_y)
            points_left.append((x, y))
        
        for i in range(len(points_left)-1):
            cv2.line(line_mask, points_left[i], points_left[i+1], intensity * 0.4, 1)
        
        # Right marionette line
        marionette_end_right_x = right_corner_x + int(w * 0.02)
        marionette_end_right_y = right_corner_y + int(h * 0.08)
        
        points_right = []
        for t in np.linspace(0, 1, 15):
            curve_offset = int(w * 0.01 * np.sin(t * np.pi))
            x = int((1-t) * right_corner_x + t * marionette_end_right_x) + curve_offset
            y = int((1-t) * right_corner_y + t * marionette_end_right_y)
            points_right.append((x, y))
        
        for i in range(len(points_right)-1):
            cv2.line(line_mask, points_right[i], points_right[i+1], intensity * 0.4, 1)
        
        result = self._apply_line_effect(result, line_mask)
        return result
    
    def _add_upper_lip_lines(self, face: np.ndarray, intensity: float) -> np.ndarray:
        """Add vertical lines above the upper lip (smoker's lines)."""
        h, w = face.shape[:2]
        result = face.copy().astype(np.float32)
        
        line_mask = np.zeros((h, w), dtype=np.float32)
        
        # Upper lip area center
        upper_lip_center_x = int(w * 0.5)
        upper_lip_y = int(h * 0.74)
        
        # Create vertical lines above upper lip
        num_lines = 5  # Central line plus 2 on each side
        line_spacing = int(w * 0.015)  # Space between lines
        
        for i in range(num_lines):
            # Calculate x position (center line at index 2)
            x_offset = (i - num_lines//2) * line_spacing
            line_x = upper_lip_center_x + x_offset
            
            # Line length varies - central lines slightly longer
            base_length = int(h * 0.025)
            if i == num_lines//2:  # Central line
                line_length = int(base_length * 1.2)
                line_intensity = intensity * 0.6
            elif abs(i - num_lines//2) == 1:  # Adjacent lines
                line_length = base_length
                line_intensity = intensity * 0.5
            else:  # Outer lines
                line_length = int(base_length * 0.8)
                line_intensity = intensity * 0.4
            
            # Draw vertical line upward from upper lip
            start_y = upper_lip_y
            end_y = upper_lip_y - line_length
            
            cv2.line(line_mask, (line_x, start_y), (line_x, end_y), line_intensity, 1)
            
            # Add slight variation to make lines look more natural
            if i % 2 == 0:  # Every other line slightly offset
                cv2.line(line_mask, (line_x-1, start_y-1), (line_x-1, end_y+1), 
                         line_intensity * 0.3, 1)
        
        result = self._apply_line_effect(result, line_mask)
        return result
    
    def _add_cheek_elevation_folds(self, face: np.ndarray, intensity: float) -> np.ndarray:
        """Add folds under eyes from cheek elevation."""
        h, w = face.shape[:2]
        result = face.copy().astype(np.float32)
        
        line_mask = np.zeros((h, w), dtype=np.float32)
        
        # Left cheek fold
        cv2.ellipse(line_mask, (int(w*0.3), int(h*0.48)), 
                   (int(w*0.06), int(h*0.015)), -20, 0, 180, intensity, 1)
        
        # Right cheek fold
        cv2.ellipse(line_mask, (int(w*0.7), int(h*0.48)),
                   (int(w*0.06), int(h*0.015)), 20, 0, 180, intensity, 1)
        
        result = self._apply_line_effect(result, line_mask)
        return result
    
    def _add_chin_fold(self, face: np.ndarray, intensity: float) -> np.ndarray:
        """Add small vertical fold in chin center."""
        h, w = face.shape[:2]
        result = face.copy().astype(np.float32)
        
        line_mask = np.zeros((h, w), dtype=np.float32)
        
        chin_x = int(w * 0.5)
        start_y = int(h * 0.85)
        end_y = int(h * 0.92)
        
        cv2.line(line_mask, (chin_x, start_y), (chin_x, end_y), intensity, 1)
        
        result = self._apply_line_effect(result, line_mask)
        return result
    
    def _add_chin_tension(self, face: np.ndarray, intensity: float) -> np.ndarray:
        """Add chin tension dimple for smiling."""
        h, w = face.shape[:2]
        result = face.copy().astype(np.float32)
        
        line_mask = np.zeros((h, w), dtype=np.float32)
        
        # Small dimple/tension mark
        cv2.circle(line_mask, (int(w*0.5), int(h*0.88)), 
                  int(w*0.01), intensity * 0.6, 1)
        
        result = self._apply_line_effect(result, line_mask)
        return result
    
    def _apply_line_effect(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply darkening effect for lines/wrinkles."""
        result = image.copy()
        
        # Darken areas where mask is applied
        darkening_factor = 0.75
        
        for c in range(3):
            channel = result[:, :, c].astype(np.float32)
            darkened = channel * darkening_factor
            result[:, :, c] = mask * darkened + (1 - mask) * channel
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _create_soft_blend_mask(self, face_shape: Tuple[int, int]) -> np.ndarray:
        """Create soft blending mask for expression lines."""
        h, w = face_shape
        
        # Create elliptical mask focusing on main facial features
        center = (w // 2, h // 2)
        axes = (w // 2 - 3, h // 2 - 3)
        
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 0.8, -1)
        
        # Soft blur for natural blending
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        
        return mask
    
    def detect_expression_type(self, face_region: np.ndarray) -> ExpressionType:
        """Automatically detect if face is smiling or neutral (basic implementation)."""
        # This is a simplified version - could be enhanced with ML models
        # For now, return neutral as default
        # In practice, you might analyze mouth curvature, eye crinkles, etc.
        return ExpressionType.NEUTRAL