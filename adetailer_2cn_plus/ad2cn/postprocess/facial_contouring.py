"""Facial contouring and makeup application system."""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional

# Optional dlib import for facial landmarks
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    dlib = None


class ContouringMask:
    """Advanced facial contouring system applying makeup guidelines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.contour_intensity = config.get('contouring', {}).get('intensity', 0.7)
        self.blend_strength = config.get('contouring', {}).get('blend_strength', 0.5)
        
        # Person-specific settings
        self.person_profiles = config.get('contouring', {}).get('person_profiles', {
            'massy': {'contour_strength': 0.8, 'expression_strength': 0.5},
            'orbi': {'contour_strength': 0.7, 'expression_strength': 0.6}, 
            'yana': {'contour_strength': 0.75, 'expression_strength': 0.55}
        })
        
        # Initialize face landmark predictor (if available)
        self.predictor = None
        self.detector = None
        
        if DLIB_AVAILABLE:
            try:
                self.detector = dlib.get_frontal_face_detector()
            except Exception:
                self.detector = None
        
    def _get_person_profile(self, image_path: Optional[str] = None) -> Dict[str, float]:
        """Get person-specific contouring settings."""
        if image_path:
            filename = image_path.lower()
            for person, profile in self.person_profiles.items():
                if person in filename:
                    return profile
        
        # Default profile
        return {'contour_strength': 0.7, 'expression_strength': 0.5}
    
    def apply_contouring(self, image: np.ndarray, face_bbox: List[int], 
                        image_path: Optional[str] = None, landmarks: Optional[Dict] = None) -> np.ndarray:
        """Apply complete facial contouring to image."""
        # Validate and refine bbox
        refined_bbox = self._refine_face_bbox(face_bbox, landmarks, image.shape[:2])
        x1, y1, x2, y2 = refined_bbox
        
        face_region = image[y1:y2, x1:x2].copy()
        
        if face_region.size == 0:
            return image
            
        profile = self._get_person_profile(image_path)
        intensity = self.contour_intensity * profile['contour_strength']
        
        # Get face landmarks if possible (fallback to dlib if no landmarks provided)
        if landmarks is None:
            landmarks = self._get_face_landmarks(face_region)
        else:
            # Adjust landmarks coordinates to face_region coordinate system
            landmarks = self._adjust_landmarks_to_face_region(landmarks, refined_bbox)
        
        # Apply contouring elements with landmark guidance
        contoured_face = self._apply_forehead_contouring(face_region, landmarks, intensity)
        contoured_face = self._apply_eyebrow_eye_contouring(contoured_face, landmarks, intensity)
        contoured_face = self._apply_nose_contouring(contoured_face, landmarks, intensity)
        contoured_face = self._apply_cheek_contouring(contoured_face, landmarks, intensity)
        contoured_face = self._apply_lip_contouring(contoured_face, landmarks, intensity)
        contoured_face = self._apply_chin_jaw_contouring(contoured_face, landmarks, intensity)
        
        # Blend back into original image
        result = image.copy()
        blend_mask = self._create_face_blend_mask(face_region.shape[:2])
        
        # Apply blending
        for c in range(3):
            face_channel = contoured_face[:, :, c].astype(np.float32)
            orig_channel = result[y1:y2, x1:x2, c].astype(np.float32)
            
            blended = blend_mask * face_channel + (1 - blend_mask) * orig_channel
            result[y1:y2, x1:x2, c] = np.clip(blended, 0, 255).astype(np.uint8)
        
        return result
    
    def _get_face_landmarks(self, face_region: np.ndarray) -> Optional[Any]:
        """Extract facial landmarks for precise positioning."""
        if not DLIB_AVAILABLE or self.detector is None:
            return None
            
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            if len(faces) > 0 and self.predictor is not None:
                return self.predictor(gray, faces[0])
        except Exception:
            pass
        return None
    
    def _apply_forehead_contouring(self, face: np.ndarray, landmarks: Optional[Any], 
                                 intensity: float) -> np.ndarray:
        """Apply forehead contouring - horizontal line + side arcs."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Horizontal line along upper forehead (darkening)
        forehead_y = int(h * 0.15)  # Upper forehead area
        forehead_mask = np.zeros((h, w), dtype=np.float32)
        
        # Create horizontal darkening line
        cv2.rectangle(forehead_mask, (int(w*0.1), forehead_y-3), 
                     (int(w*0.9), forehead_y+3), intensity * 0.3, -1)
        
        # Vertical arcs on sides (narrowing)
        left_arc_x = int(w * 0.08)
        right_arc_x = int(w * 0.92)
        
        for y in range(int(h*0.1), int(h*0.4)):
            arc_intensity = intensity * 0.25 * (1 - abs(y - h*0.25) / (h*0.15))
            if arc_intensity > 0:
                forehead_mask[y, left_arc_x-2:left_arc_x+2] = arc_intensity
                forehead_mask[y, right_arc_x-2:right_arc_x+2] = arc_intensity
        
        # Apply darkening
        result = self._apply_darkening_mask(result, forehead_mask)
        return result
    
    def _apply_eyebrow_eye_contouring(self, face: np.ndarray, landmarks: Optional[Any],
                                    intensity: float) -> np.ndarray:
        """Apply eyebrow and eye area contouring."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Under eyebrows - light highlight
        brow_highlight_mask = np.zeros((h, w), dtype=np.float32)
        
        left_brow_y = int(h * 0.32)
        right_brow_y = int(h * 0.32)
        
        # Left eyebrow highlight
        cv2.ellipse(brow_highlight_mask, (int(w*0.27), left_brow_y), 
                   (int(w*0.06), int(h*0.02)), 0, 0, 360, intensity * 0.4, -1)
        
        # Right eyebrow highlight  
        cv2.ellipse(brow_highlight_mask, (int(w*0.73), right_brow_y),
                   (int(w*0.06), int(h*0.02)), 0, 0, 360, intensity * 0.4, -1)
        
        # Above eyebrows outside - darkening for curve accent
        brow_shadow_mask = np.zeros((h, w), dtype=np.float32)
        
        cv2.ellipse(brow_shadow_mask, (int(w*0.22), int(h*0.28)),
                   (int(w*0.03), int(h*0.015)), 0, 0, 360, intensity * 0.2, -1)
        cv2.ellipse(brow_shadow_mask, (int(w*0.78), int(h*0.28)),
                   (int(w*0.03), int(h*0.015)), 0, 0, 360, intensity * 0.2, -1)
        
        # Under lower eyelid highlight (inner corners)
        eye_highlight_mask = np.zeros((h, w), dtype=np.float32)
        
        cv2.circle(eye_highlight_mask, (int(w*0.32), int(h*0.42)), 
                  int(w*0.015), intensity * 0.3, -1)
        cv2.circle(eye_highlight_mask, (int(w*0.68), int(h*0.42)),
                  int(w*0.015), intensity * 0.3, -1)
        
        # Apply highlights and shadows
        result = self._apply_highlighting_mask(result, brow_highlight_mask)
        result = self._apply_highlighting_mask(result, eye_highlight_mask)
        result = self._apply_darkening_mask(result, brow_shadow_mask)
        
        return result
    
    def _apply_nose_contouring(self, face: np.ndarray, landmarks: Optional[Any],
                              intensity: float) -> np.ndarray:
        """Apply nose contouring - parallel lines + highlights."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Two parallel vertical lines along nose bridge (darkening)
        nose_shadow_mask = np.zeros((h, w), dtype=np.float32)
        
        nose_center_x = int(w * 0.5)
        nose_width = int(w * 0.015)
        nose_start_y = int(h * 0.35)
        nose_end_y = int(h * 0.62)
        
        # Left shadow line
        cv2.rectangle(nose_shadow_mask, 
                     (nose_center_x - nose_width - 2, nose_start_y),
                     (nose_center_x - nose_width, nose_end_y),
                     intensity * 0.3, -1)
        
        # Right shadow line  
        cv2.rectangle(nose_shadow_mask,
                     (nose_center_x + nose_width, nose_start_y),
                     (nose_center_x + nose_width + 2, nose_end_y),
                     intensity * 0.3, -1)
        
        # Nose bridge highlight (central strip)
        nose_highlight_mask = np.zeros((h, w), dtype=np.float32)
        
        cv2.rectangle(nose_highlight_mask,
                     (nose_center_x - nose_width//2, nose_start_y),
                     (nose_center_x + nose_width//2, nose_end_y),
                     intensity * 0.4, -1)
        
        # Nose tip highlight
        cv2.circle(nose_highlight_mask, (nose_center_x, int(h * 0.58)),
                  int(w * 0.012), intensity * 0.35, -1)
        
        # Apply contouring
        result = self._apply_darkening_mask(result, nose_shadow_mask)
        result = self._apply_highlighting_mask(result, nose_highlight_mask)
        
        return result
    
    def _apply_cheek_contouring(self, face: np.ndarray, landmarks: Optional[Any],
                               intensity: float) -> np.ndarray:
        """Apply cheek contouring - diagonal shadow lines + highlights."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Diagonal shadow lines from ear to mouth corner (below cheekbones)
        cheek_shadow_mask = np.zeros((h, w), dtype=np.float32)
        
        # Left cheek shadow
        left_pts = np.array([
            [int(w*0.05), int(h*0.55)],  # Near ear
            [int(w*0.15), int(h*0.60)],
            [int(w*0.25), int(h*0.68)],
            [int(w*0.35), int(h*0.75)],  # Toward mouth
            [int(w*0.30), int(h*0.80)],
            [int(w*0.10), int(h*0.65)]
        ], np.int32)
        
        cv2.fillPoly(cheek_shadow_mask, [left_pts], intensity * 0.25)
        
        # Right cheek shadow
        right_pts = np.array([
            [int(w*0.95), int(h*0.55)],  # Near ear
            [int(w*0.85), int(h*0.60)],
            [int(w*0.75), int(h*0.68)],
            [int(w*0.65), int(h*0.75)],  # Toward mouth
            [int(w*0.70), int(h*0.80)],
            [int(w*0.90), int(h*0.65)]
        ], np.int32)
        
        cv2.fillPoly(cheek_shadow_mask, [right_pts], intensity * 0.25)
        
        # Cheekbone highlights (above shadow lines)
        cheek_highlight_mask = np.zeros((h, w), dtype=np.float32)
        
        # Left cheek highlight
        cv2.ellipse(cheek_highlight_mask, (int(w*0.25), int(h*0.52)),
                   (int(w*0.08), int(h*0.03)), -30, 0, 360, intensity * 0.4, -1)
        
        # Right cheek highlight
        cv2.ellipse(cheek_highlight_mask, (int(w*0.75), int(h*0.52)),
                   (int(w*0.08), int(h*0.03)), 30, 0, 360, intensity * 0.4, -1)
        
        # Apply contouring
        result = self._apply_darkening_mask(result, cheek_shadow_mask)
        result = self._apply_highlighting_mask(result, cheek_highlight_mask)
        
        return result
    
    def _apply_lip_contouring(self, face: np.ndarray, landmarks: Optional[Any],
                            intensity: float) -> np.ndarray:
        """Apply enhanced lip contouring with landmark precision."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Get lip positioning - use landmarks if available, otherwise estimate
        lip_coords = self._get_lip_coordinates(landmarks, face.shape)
        
        if lip_coords:
            result = self._apply_landmark_based_lip_contouring(result, lip_coords, intensity)
        else:
            result = self._apply_estimated_lip_contouring(result, intensity)
        
        # Add perioral area enhancements
        result = self._apply_perioral_enhancements(result, lip_coords, intensity)
        
        return result
    
    def _get_lip_coordinates(self, landmarks: Optional[Any], face_shape: tuple) -> Optional[Dict]:
        """Extract or estimate lip coordinates."""
        if landmarks and isinstance(landmarks, dict) and 'mouth' in landmarks:
            try:
                mouth_points = np.array(landmarks['mouth'])
                if len(mouth_points) > 0:
                    # MediaPipe mouth landmarks mapping
                    return {
                        'upper_lip_center': mouth_points[0] if len(mouth_points) > 0 else None,
                        'lower_lip_center': mouth_points[-1] if len(mouth_points) > 0 else None,
                        'left_corner': mouth_points[len(mouth_points)//4] if len(mouth_points) > 4 else None,
                        'right_corner': mouth_points[3*len(mouth_points)//4] if len(mouth_points) > 4 else None,
                        'mouth_center': np.mean(mouth_points, axis=0).astype(int),
                        'mouth_width': np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0]),
                        'mouth_height': np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1])
                    }
            except Exception:
                pass
        
        # Fallback to estimated positions
        h, w = face_shape[:2]
        return {
            'upper_lip_center': [int(w*0.5), int(h*0.74)],
            'lower_lip_center': [int(w*0.5), int(h*0.78)],
            'left_corner': [int(w*0.42), int(h*0.76)],
            'right_corner': [int(w*0.58), int(h*0.76)],
            'mouth_center': [int(w*0.5), int(h*0.76)],
            'mouth_width': int(w*0.16),
            'mouth_height': int(h*0.04)
        }
    
    def _apply_landmark_based_lip_contouring(self, face: np.ndarray, lip_coords: Dict, 
                                           intensity: float) -> np.ndarray:
        """Apply precise lip contouring using landmark coordinates."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Enhanced Cupid's bow with precise positioning
        lip_highlight_mask = np.zeros((h, w), dtype=np.float32)
        
        upper_lip = lip_coords['upper_lip_center']
        mouth_width = lip_coords['mouth_width']
        
        # Detailed Cupid's bow - central dip and side peaks
        cupid_bow_center = upper_lip
        peak_offset = max(4, int(mouth_width * 0.15))
        
        # Central dip (deeper highlight)
        cv2.circle(lip_highlight_mask, tuple(cupid_bow_center),
                  max(2, int(mouth_width * 0.06)), intensity * 0.4, -1)
        
        # Side peaks (more prominent)
        left_peak = [cupid_bow_center[0] - peak_offset, cupid_bow_center[1] - 1]
        right_peak = [cupid_bow_center[0] + peak_offset, cupid_bow_center[1] - 1]
        
        cv2.circle(lip_highlight_mask, tuple(left_peak),
                  max(2, int(mouth_width * 0.05)), intensity * 0.35, -1)
        cv2.circle(lip_highlight_mask, tuple(right_peak),
                  max(2, int(mouth_width * 0.05)), intensity * 0.35, -1)
        
        # Upper lip contour line
        self._draw_lip_contour_line(lip_highlight_mask, upper_lip, mouth_width, 
                                   intensity * 0.25, is_upper=True)
        
        # Lower lip volume enhancement
        lower_lip = lip_coords['lower_lip_center']
        cv2.ellipse(lip_highlight_mask, tuple(lower_lip),
                   (max(4, int(mouth_width * 0.4)), max(2, int(mouth_width * 0.15))),
                   0, 0, 360, intensity * 0.3, -1)
        
        # Lip shadows and definition
        lip_shadow_mask = np.zeros((h, w), dtype=np.float32)
        
        # Under lower lip shadow (more realistic shape)
        shadow_center = [lower_lip[0], lower_lip[1] + max(3, int(mouth_width * 0.2))]
        cv2.ellipse(lip_shadow_mask, tuple(shadow_center),
                   (max(5, int(mouth_width * 0.5)), max(2, int(mouth_width * 0.1))),
                   0, 0, 360, intensity * 0.25, -1)
        
        # Corner shadows for definition
        left_corner = lip_coords['left_corner']
        right_corner = lip_coords['right_corner']
        
        corner_shadow_size = max(2, int(mouth_width * 0.08))
        cv2.circle(lip_shadow_mask, tuple(left_corner),
                  corner_shadow_size, intensity * 0.15, -1)
        cv2.circle(lip_shadow_mask, tuple(right_corner),
                  corner_shadow_size, intensity * 0.15, -1)
        
        # Apply all enhancements
        result = self._apply_highlighting_mask(result, lip_highlight_mask)
        result = self._apply_darkening_mask(result, lip_shadow_mask)
        
        return result
    
    def _apply_estimated_lip_contouring(self, face: np.ndarray, intensity: float) -> np.ndarray:
        """Apply basic lip contouring when landmarks not available."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Enhanced version of original basic contouring
        lip_highlight_mask = np.zeros((h, w), dtype=np.float32)
        
        cupids_bow_y = int(h * 0.75)
        mouth_width_est = int(w * 0.16)
        
        # More detailed Cupid's bow
        cv2.circle(lip_highlight_mask, (int(w*0.5), cupids_bow_y),
                  max(2, int(w*0.01)), intensity * 0.35, -1)
        
        # Enhanced side peaks
        peak_offset = max(4, int(mouth_width_est * 0.15))
        cv2.circle(lip_highlight_mask, (int(w*0.5) - peak_offset, cupids_bow_y-1),
                  max(2, int(w*0.008)), intensity * 0.3, -1)
        cv2.circle(lip_highlight_mask, (int(w*0.5) + peak_offset, cupids_bow_y-1),
                  max(2, int(w*0.008)), intensity * 0.3, -1)
        
        # Lower lip volume
        cv2.ellipse(lip_highlight_mask, (int(w*0.5), int(h*0.78)),
                   (max(4, int(w*0.06)), max(2, int(w*0.02))), 0, 0, 360, intensity * 0.25, -1)
        
        # Enhanced shadows
        lip_shadow_mask = np.zeros((h, w), dtype=np.float32)
        
        cv2.ellipse(lip_shadow_mask, (int(w*0.5), int(h*0.82)),
                   (max(4, int(w*0.08)), max(2, int(w*0.02))), 0, 0, 360, intensity * 0.2, -1)
        
        # Apply enhancements
        result = self._apply_highlighting_mask(result, lip_highlight_mask)
        result = self._apply_darkening_mask(result, lip_shadow_mask)
        
        return result
    
    def _draw_lip_contour_line(self, mask: np.ndarray, center_point: List[int], 
                              mouth_width: int, intensity: float, is_upper: bool = True):
        """Draw realistic lip contour line."""
        center_x, center_y = center_point
        half_width = mouth_width // 2
        
        # Create curved contour line
        points = []
        for i in range(-half_width, half_width + 1, 2):
            x = center_x + i
            # Create natural lip curve
            if is_upper:
                # Upper lip has M-shape (Cupid's bow)
                curve_factor = abs(i) / half_width
                y_offset = int(2 * curve_factor * (1 - curve_factor))  # Parabolic curve
                y = center_y + y_offset
            else:
                # Lower lip has gentle curve
                curve_factor = (i / half_width) ** 2
                y_offset = int(1.5 * curve_factor)
                y = center_y + y_offset
            
            points.append([x, y])
        
        # Draw contour line
        for i in range(len(points) - 1):
            cv2.line(mask, tuple(points[i]), tuple(points[i + 1]), intensity, 1)
    
    def _apply_perioral_enhancements(self, face: np.ndarray, lip_coords: Optional[Dict], 
                                   intensity: float) -> np.ndarray:
        """Add perioral area enhancements (around mouth area)."""
        if not lip_coords:
            return face
            
        h, w = face.shape[:2]
        result = face.copy()
        
        # Philtrum enhancement (area between nose and upper lip)
        philtrum_mask = np.zeros((h, w), dtype=np.float32)
        
        upper_lip = lip_coords['upper_lip_center']
        philtrum_center = [upper_lip[0], upper_lip[1] - max(5, int(lip_coords['mouth_height'] * 1.2))]
        
        # Subtle philtrum lines
        cv2.line(philtrum_mask, 
                (philtrum_center[0] - 1, philtrum_center[1]), 
                (upper_lip[0] - 2, upper_lip[1]), 
                intensity * 0.1, 1)
        cv2.line(philtrum_mask, 
                (philtrum_center[0] + 1, philtrum_center[1]), 
                (upper_lip[0] + 2, upper_lip[1]), 
                intensity * 0.1, 1)
        
        # Lip corners enhancement for more definition
        corner_enhance_mask = np.zeros((h, w), dtype=np.float32)
        
        left_corner = lip_coords['left_corner']
        right_corner = lip_coords['right_corner']
        
        # Small corner shadows for definition
        corner_size = max(1, int(lip_coords['mouth_width'] * 0.05))
        cv2.circle(corner_enhance_mask, tuple(left_corner), corner_size, intensity * 0.12, -1)
        cv2.circle(corner_enhance_mask, tuple(right_corner), corner_size, intensity * 0.12, -1)
        
        # Apply perioral enhancements
        result = self._apply_darkening_mask(result, philtrum_mask)
        result = self._apply_darkening_mask(result, corner_enhance_mask)
        
        return result
    
    def _apply_chin_jaw_contouring(self, face: np.ndarray, landmarks: Optional[Any],
                                  intensity: float) -> np.ndarray:
        """Apply chin and jawline contouring."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Under chin darkening
        chin_shadow_mask = np.zeros((h, w), dtype=np.float32)
        
        cv2.ellipse(chin_shadow_mask, (int(w*0.5), int(h*0.95)),
                   (int(w*0.12), int(h*0.08)), 0, 0, 360, intensity * 0.25, -1)
        
        # Jawline darkening
        jaw_shadow_mask = np.zeros((h, w), dtype=np.float32)
        
        # Left jawline
        left_jaw_pts = np.array([
            [int(w*0.05), int(h*0.75)],
            [int(w*0.15), int(h*0.85)],
            [int(w*0.35), int(h*0.92)],
            [int(w*0.30), int(h*0.95)],
            [int(w*0.10), int(h*0.90)],
            [int(w*0.02), int(h*0.80)]
        ], np.int32)
        
        cv2.fillPoly(jaw_shadow_mask, [left_jaw_pts], intensity * 0.2)
        
        # Right jawline
        right_jaw_pts = np.array([
            [int(w*0.95), int(h*0.75)],
            [int(w*0.85), int(h*0.85)],
            [int(w*0.65), int(h*0.92)],
            [int(w*0.70), int(h*0.95)],
            [int(w*0.90), int(h*0.90)],
            [int(w*0.98), int(h*0.80)]
        ], np.int32)
        
        cv2.fillPoly(jaw_shadow_mask, [right_jaw_pts], intensity * 0.2)
        
        # Center chin highlight
        chin_highlight_mask = np.zeros((h, w), dtype=np.float32)
        
        cv2.circle(chin_highlight_mask, (int(w*0.5), int(h*0.88)),
                  int(w*0.015), intensity * 0.3, -1)
        
        # Apply contouring
        result = self._apply_darkening_mask(result, chin_shadow_mask)
        result = self._apply_darkening_mask(result, jaw_shadow_mask)
        result = self._apply_highlighting_mask(result, chin_highlight_mask)
        
        return result
    
    def _apply_darkening_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply advanced darkening effect with gradient blending and color adaptation."""
        result = image.copy().astype(np.float32)
        
        # Analyze skin tone for adaptive color mixing
        skin_tone = self._analyze_skin_tone(result)
        
        # Create gradient mask for smoother transitions
        gradient_mask = self._create_gradient_mask(mask)
        
        # Adaptive darkening based on skin tone
        darkening_factor = self._get_adaptive_darkening_factor(skin_tone)
        
        # Apply contouring with color-aware mixing
        for c in range(3):
            channel = result[:, :, c]
            
            # Create natural shadow color (not just darker, but cooler)
            shadow_color = self._create_shadow_color(channel, skin_tone, c)
            
            # Apply with gradient blending
            result[:, :, c] = gradient_mask * shadow_color + (1 - gradient_mask) * channel
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_highlighting_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply advanced highlighting effect with gradient blending and color adaptation."""
        result = image.copy().astype(np.float32)
        
        # Analyze skin tone for adaptive highlighting
        skin_tone = self._analyze_skin_tone(result)
        
        # Create gradient mask for smoother transitions
        gradient_mask = self._create_gradient_mask(mask)
        
        # Adaptive highlighting based on skin tone
        highlighting_factor = self._get_adaptive_highlighting_factor(skin_tone)
        
        # Apply highlighting with color-aware mixing
        for c in range(3):
            channel = result[:, :, c]
            
            # Create natural highlight color (warmer and brighter)
            highlight_color = self._create_highlight_color(channel, skin_tone, c)
            
            # Apply with gradient blending
            result[:, :, c] = gradient_mask * highlight_color + (1 - gradient_mask) * channel
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _analyze_skin_tone(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze skin tone characteristics for adaptive makeup."""
        # Convert to different color spaces for analysis
        lab_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB)
        
        # Calculate average skin tone values
        l_mean = np.mean(lab_image[:, :, 0])  # Lightness
        a_mean = np.mean(lab_image[:, :, 1])  # Green-Red axis
        b_mean = np.mean(lab_image[:, :, 2])  # Blue-Yellow axis
        
        # Determine skin characteristics
        is_warm_tone = b_mean > 128  # Higher B values indicate warmer (yellow) tones
        is_light_skin = l_mean > 140  # Higher L values indicate lighter skin
        
        return {
            'lightness': l_mean,
            'warmth': b_mean - 128,  # Positive = warm, negative = cool
            'saturation': a_mean - 128,  # Green-red saturation
            'is_warm': is_warm_tone,
            'is_light': is_light_skin
        }
    
    def _create_gradient_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create gradient version of mask for smoother blending."""
        # Apply multiple levels of gaussian blur for gradient effect
        gradient_mask = mask.copy().astype(np.float32)
        
        # Create multiple layers of falloff
        blur_sizes = [3, 5, 9]
        weights = [0.6, 0.3, 0.1]
        
        blended_mask = np.zeros_like(gradient_mask)
        
        for blur_size, weight in zip(blur_sizes, weights):
            blurred = cv2.GaussianBlur(gradient_mask, (blur_size, blur_size), 0)
            blended_mask += blurred * weight
        
        # Normalize and apply power curve for more natural falloff
        blended_mask = blended_mask / np.sum(weights)
        blended_mask = np.power(blended_mask, 1.5)  # Power curve for softer edges
        
        return blended_mask
    
    def _get_adaptive_darkening_factor(self, skin_tone: Dict[str, float]) -> float:
        """Calculate adaptive darkening factor based on skin tone."""
        base_factor = 0.75
        
        # Lighter skin needs less darkening to avoid harsh contrast
        if skin_tone['is_light']:
            base_factor = 0.82
        else:
            base_factor = 0.68
        
        # Adjust based on warmth (warm tones can handle slightly more contrast)
        if skin_tone['is_warm']:
            base_factor *= 0.95
        else:
            base_factor *= 1.05
        
        return base_factor
    
    def _get_adaptive_highlighting_factor(self, skin_tone: Dict[str, float]) -> float:
        """Calculate adaptive highlighting factor based on skin tone."""
        base_factor = 1.25
        
        # Light skin needs gentler highlighting
        if skin_tone['is_light']:
            base_factor = 1.2
        else:
            base_factor = 1.35
        
        # Warm tones can handle more highlighting
        if skin_tone['is_warm']:
            base_factor *= 1.1
        else:
            base_factor *= 0.95
        
        return base_factor
    
    def _create_shadow_color(self, channel: np.ndarray, skin_tone: Dict[str, float], 
                           color_channel: int) -> np.ndarray:
        """Create natural shadow color that's not just darker."""
        # Start with darkened base
        darkening_factor = self._get_adaptive_darkening_factor(skin_tone)
        shadow_base = channel * darkening_factor
        
        # Add color temperature adjustment for natural shadows
        if color_channel == 0:  # Blue channel
            # Shadows tend to be slightly cooler (more blue)
            if skin_tone['is_warm']:
                shadow_base = shadow_base * 1.02  # Slight blue boost for warm skin
        elif color_channel == 1:  # Green channel
            # Neutral adjustment
            pass
        elif color_channel == 2:  # Red channel
            # Shadows have less red/warmth
            shadow_base = shadow_base * 0.98
        
        return shadow_base
    
    def _create_highlight_color(self, channel: np.ndarray, skin_tone: Dict[str, float], 
                              color_channel: int) -> np.ndarray:
        """Create natural highlight color that's warmer and brighter."""
        # Start with brightened base
        highlighting_factor = self._get_adaptive_highlighting_factor(skin_tone)
        highlight_base = np.minimum(channel * highlighting_factor, 255)
        
        # Add warmth to highlights
        if color_channel == 0:  # Blue channel
            # Highlights have less blue (warmer)
            highlight_base = highlight_base * 0.98
        elif color_channel == 1:  # Green channel
            # Slight boost for natural skin glow
            if skin_tone['is_warm']:
                highlight_base = highlight_base * 1.01
        elif color_channel == 2:  # Red channel
            # Highlights are warmer (more red/yellow)
            highlight_base = highlight_base * 1.03
        
        return highlight_base
    
    def _create_face_blend_mask(self, face_shape: Tuple[int, int]) -> np.ndarray:
        """Create smooth blending mask for face region."""
        h, w = face_shape
        
        # Create elliptical mask
        center = (w // 2, h // 2)
        axes = (w // 2 - 5, h // 2 - 5)
        
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, center, axes, 0, 0, 360, self.blend_strength, -1)
        
        # Apply Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def _refine_face_bbox(self, face_bbox: List[int], landmarks: Optional[Dict], 
                         image_shape: Tuple[int, int]) -> List[int]:
        """Refine face bbox using landmarks or validation rules.
        
        Args:
            face_bbox: Original bounding box [x1, y1, x2, y2]
            landmarks: MediaPipe landmarks dict (optional)
            image_shape: Image shape (h, w)
            
        Returns:
            Refined bounding box [x1, y1, x2, y2]
        """
        h, w = image_shape
        x1, y1, x2, y2 = face_bbox
        
        # If we have MediaPipe landmarks, use them for precise bbox
        if landmarks and isinstance(landmarks, dict) and 'face_oval' in landmarks:
            try:
                face_oval_points = np.array(landmarks['face_oval'])
                
                if len(face_oval_points) > 0:
                    x_coords = face_oval_points[:, 0]
                    y_coords = face_oval_points[:, 1]
                    
                    refined_x1 = int(np.min(x_coords))
                    refined_y1 = int(np.min(y_coords))
                    refined_x2 = int(np.max(x_coords))
                    refined_y2 = int(np.max(y_coords))
                    
                    # Add smart padding (15% buffer)
                    face_width = refined_x2 - refined_x1
                    face_height = refined_y2 - refined_y1
                    
                    padding_x = int(face_width * 0.15)
                    padding_y = int(face_height * 0.15)
                    
                    refined_x1 = max(0, refined_x1 - padding_x)
                    refined_y1 = max(0, refined_y1 - padding_y)
                    refined_x2 = min(w, refined_x2 + padding_x)
                    refined_y2 = min(h, refined_y2 + padding_y)
                    
                    # Validate the refined bbox
                    if self._validate_bbox([refined_x1, refined_y1, refined_x2, refined_y2], image_shape):
                        return [refined_x1, refined_y1, refined_x2, refined_y2]
            except Exception:
                pass  # Fall through to original bbox validation
        
        # Validate original bbox and apply basic refinements
        validated_bbox = self._validate_and_fix_bbox(face_bbox, image_shape)
        return validated_bbox if validated_bbox else face_bbox
    
    def _validate_bbox(self, bbox: List[int], image_shape: Tuple[int, int]) -> bool:
        """Validate bounding box dimensions and properties.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            image_shape: Image shape (h, w)
            
        Returns:
            True if bbox is valid
        """
        h, w = image_shape
        x1, y1, x2, y2 = bbox
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
            return False
        
        # Check positive dimensions
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Check minimum size (face should be at least 32x32)
        if (x2 - x1) < 32 or (y2 - y1) < 32:
            return False
        
        # Check maximum size (face shouldn't be more than 80% of image)
        if (x2 - x1) > w * 0.8 or (y2 - y1) > h * 0.8:
            return False
        
        # Check aspect ratio (face should be roughly 0.7 to 1.3 ratio)
        aspect_ratio = (x2 - x1) / (y2 - y1)
        if aspect_ratio < 0.6 or aspect_ratio > 1.4:
            return False
        
        return True
    
    def _validate_and_fix_bbox(self, bbox: List[int], image_shape: Tuple[int, int]) -> List[int]:
        """Validate and fix bounding box if possible.
        
        Args:
            bbox: Original bounding box [x1, y1, x2, y2]
            image_shape: Image shape (h, w)
            
        Returns:
            Fixed bounding box or original if already valid
        """
        h, w = image_shape
        x1, y1, x2, y2 = bbox
        
        # Fix bounds
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Check if bbox is too large (more than 70% of image)
        face_width = x2 - x1
        face_height = y2 - y1
        
        if face_width > w * 0.7 or face_height > h * 0.7:
            # Shrink bbox to reasonable size
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            max_width = int(w * 0.6)
            max_height = int(h * 0.6)
            
            new_width = min(face_width, max_width)
            new_height = min(face_height, max_height)
            
            x1 = max(0, center_x - new_width // 2)
            y1 = max(0, center_y - new_height // 2)
            x2 = min(w, x1 + new_width)
            y2 = min(h, y1 + new_height)
        
        # Ensure minimum size
        if face_width < 64:
            center_x = (x1 + x2) // 2
            x1 = max(0, center_x - 32)
            x2 = min(w, center_x + 32)
            
        if face_height < 64:
            center_y = (y1 + y2) // 2
            y1 = max(0, center_y - 32)
            y2 = min(h, center_y + 32)
        
        return [int(x1), int(y1), int(x2), int(y2)]
    
    def _adjust_landmarks_to_face_region(self, landmarks: Dict, refined_bbox: List[int]) -> Optional[Dict]:
        """Adjust landmark coordinates to face region coordinate system.
        
        Args:
            landmarks: MediaPipe landmarks dict with absolute coordinates
            refined_bbox: Face region bbox [x1, y1, x2, y2]
            
        Returns:
            Adjusted landmarks dict with relative coordinates to face region
        """
        if not landmarks or not isinstance(landmarks, dict):
            return None
        
        x1, y1, x2, y2 = refined_bbox
        adjusted_landmarks = {}
        
        try:
            for region_name, points in landmarks.items():
                if not points:
                    continue
                    
                adjusted_points = []
                for point in points:
                    if len(point) >= 2:
                        # Convert from absolute image coordinates to face region coordinates
                        rel_x = point[0] - x1
                        rel_y = point[1] - y1
                        adjusted_points.append([rel_x, rel_y])
                
                if adjusted_points:
                    adjusted_landmarks[region_name] = adjusted_points
            
            return adjusted_landmarks if adjusted_landmarks else None
            
        except Exception:
            return None