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
        # DRASTICALLY REDUCED intensities for natural look
        self.contour_intensity = config.get('contouring', {}).get('intensity', 0.15)  # Was 0.7, now 0.15
        self.blend_strength = config.get('contouring', {}).get('blend_strength', 0.8)  # Higher blend for smoother transitions
        
        # Natural makeup intensity limits
        self.max_highlight_intensity = 0.12  # Maximum highlight strength
        self.max_shadow_intensity = 0.08     # Maximum shadow strength
        
        # Person-specific settings with NATURAL intensities
        self.person_profiles = config.get('contouring', {}).get('person_profiles', {
            'massy': {'contour_strength': 0.12, 'expression_strength': 0.08},  # Was 0.8/0.5
            'orbi': {'contour_strength': 0.10, 'expression_strength': 0.09},   # Was 0.7/0.6
            'yana': {'contour_strength': 0.11, 'expression_strength': 0.085}   # Was 0.75/0.55
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
        
        # Default profile with NATURAL intensities
        return {'contour_strength': 0.10, 'expression_strength': 0.08}  # Was 0.7/0.5
    
    def apply_contouring(self, image: np.ndarray, face_bbox: List[int], 
                        image_path: Optional[str] = None, landmarks: Optional[Dict] = None) -> np.ndarray:
        """Apply complete NATURAL facial enhancement with validation."""
        # Validate input image first
        if not self._validate_input_image(image):
            print("Input validation failed - returning original image")
            return image
            
        # Validate and refine bbox
        refined_bbox = self._refine_face_bbox(face_bbox, landmarks, image.shape[:2])
        x1, y1, x2, y2 = refined_bbox
        
        face_region = image[y1:y2, x1:x2].copy()
        
        if face_region.size == 0:
            print("Empty face region - returning original image")
            return image
        
        # VALIDATE face region quality
        if not self._validate_face_region(face_region):
            print("Face region validation failed - returning original image")
            return image
            
        profile = self._get_person_profile(image_path)
        base_intensity = self.contour_intensity * profile['contour_strength']
        
        # APPLY SAFETY LIMITS to prevent unnatural results
        safe_intensity = self._apply_safety_limits(base_intensity, face_region)
        
        # Store current face for color analysis
        self._current_face = face_region.copy()
        
        # Get face landmarks if possible
        if landmarks is None:
            landmarks = self._get_face_landmarks(face_region)
        else:
            landmarks = self._adjust_landmarks_to_face_region(landmarks, refined_bbox)
        
        # Apply natural contouring with validation at each step
        original_face = face_region.copy()
        contoured_face = face_region.copy()
        
        try:
            # Apply each enhancement with individual validation
            contoured_face = self._apply_forehead_contouring(contoured_face, landmarks, safe_intensity)
            contoured_face = self._validate_step_result(contoured_face, original_face, "forehead")
            
            contoured_face = self._apply_eyebrow_eye_contouring(contoured_face, landmarks, safe_intensity)
            contoured_face = self._validate_step_result(contoured_face, original_face, "eyebrow")
            
            contoured_face = self._apply_nose_contouring(contoured_face, landmarks, safe_intensity)
            contoured_face = self._validate_step_result(contoured_face, original_face, "nose")
            
            contoured_face = self._apply_cheek_contouring(contoured_face, landmarks, safe_intensity)
            contoured_face = self._validate_step_result(contoured_face, original_face, "cheek")
            
            contoured_face = self._apply_lip_contouring(contoured_face, landmarks, safe_intensity)
            contoured_face = self._validate_step_result(contoured_face, original_face, "lip")
            
            contoured_face = self._apply_chin_jaw_contouring(contoured_face, landmarks, safe_intensity)
            contoured_face = self._validate_step_result(contoured_face, original_face, "chin_jaw")
            
        except Exception as e:
            print(f"Contouring error: {e} - returning original image")
            return image
        
        # Final validation of complete result
        if not self._validate_final_result(contoured_face, original_face):
            print("Final validation failed - returning original image")
            return image
        
        # Clean up
        if hasattr(self, '_current_face'):
            delattr(self, '_current_face')
        
        # Apply natural blending back to original image
        result = self._blend_face_naturally(image, contoured_face, refined_bbox)
        
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
        """Apply NATURAL forehead contouring using organic shapes."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Create natural forehead shadow mask
        forehead_mask = np.zeros((h, w), dtype=np.float32)
        
        # NATURAL forehead positioning - adapt to face proportions
        face_ratio = w / h
        if face_ratio > 0.9:  # Wide face
            forehead_y = int(h * 0.12)
            side_margin = int(w * 0.15)
        elif face_ratio < 0.75:  # Narrow face
            forehead_y = int(h * 0.10)
            side_margin = int(w * 0.12)
        else:  # Normal proportions
            forehead_y = int(h * 0.11)
            side_margin = int(w * 0.13)
        
        # Create NATURAL curved forehead shadow (not straight line)
        center_x = w // 2
        shadow_width = w - 2 * side_margin
        
        # Draw natural curved shadow using ellipse instead of rectangle
        cv2.ellipse(forehead_mask, (center_x, forehead_y), 
                   (shadow_width // 2, int(h * 0.015)), 0, 0, 360, 
                   intensity * self.max_shadow_intensity, -1)
        
        # Add subtle temple shadows (organic curves, not straight lines)
        temple_y_start = int(h * 0.08)
        temple_y_end = int(h * 0.35)
        
        # Left temple shadow - natural curve
        for y in range(temple_y_start, temple_y_end):
            progress = (y - temple_y_start) / (temple_y_end - temple_y_start)
            curve_intensity = intensity * self.max_shadow_intensity * 0.6 * (1 - progress) * np.sin(progress * np.pi)
            
            if curve_intensity > 0.01:  # Only apply if visible
                temple_x = int(side_margin * 0.7 + progress * side_margin * 0.3)
                cv2.circle(forehead_mask, (temple_x, y), 
                          max(2, int(w * 0.008)), curve_intensity, -1)
        
        # Right temple shadow - natural curve
        for y in range(temple_y_start, temple_y_end):
            progress = (y - temple_y_start) / (temple_y_end - temple_y_start)
            curve_intensity = intensity * self.max_shadow_intensity * 0.6 * (1 - progress) * np.sin(progress * np.pi)
            
            if curve_intensity > 0.01:  # Only apply if visible
                temple_x = int(w - side_margin * 0.7 - progress * side_margin * 0.3)
                cv2.circle(forehead_mask, (temple_x, y), 
                          max(2, int(w * 0.008)), curve_intensity, -1)
        
        # Apply natural darkening
        result = self._apply_darkening_mask(result, forehead_mask)
        return result
    
    def _apply_eyebrow_eye_contouring(self, face: np.ndarray, landmarks: Optional[Any],
                                    intensity: float) -> np.ndarray:
        """Apply NATURAL eyebrow and eye area enhancement."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Adaptive positioning based on face proportions
        face_ratio = w / h
        if face_ratio > 0.9:  # Wide face
            eye_y_ratio = 0.35
            eye_spacing = 0.46
            brow_y_offset = 0.04
        elif face_ratio < 0.75:  # Narrow face
            eye_y_ratio = 0.32
            eye_spacing = 0.42
            brow_y_offset = 0.035
        else:  # Normal proportions
            eye_y_ratio = 0.34
            eye_spacing = 0.44
            brow_y_offset = 0.038
        
        # NATURAL under-eyebrow highlighting
        brow_highlight_mask = np.zeros((h, w), dtype=np.float32)
        
        left_brow_center = (int(w * (0.5 - eye_spacing/2)), int(h * (eye_y_ratio - brow_y_offset)))
        right_brow_center = (int(w * (0.5 + eye_spacing/2)), int(h * (eye_y_ratio - brow_y_offset)))
        
        # Natural brow highlight shapes (follow brow arch)
        highlight_intensity = intensity * self.max_highlight_intensity * 2.0  # Still subtle
        
        # Left eyebrow highlight with natural arch shape
        cv2.ellipse(brow_highlight_mask, left_brow_center, 
                   (max(8, int(w*0.04)), max(3, int(h*0.012))), 
                   -15, 0, 360, highlight_intensity, -1)  # Slight angle for natural arch
        
        # Right eyebrow highlight with natural arch shape
        cv2.ellipse(brow_highlight_mask, right_brow_center,
                   (max(8, int(w*0.04)), max(3, int(h*0.012))), 
                   15, 0, 360, highlight_intensity, -1)   # Opposite angle for natural arch
        
        # VERY SUBTLE brow bone enhancement (no harsh shadows)
        brow_shadow_mask = np.zeros((h, w), dtype=np.float32)
        
        shadow_intensity = intensity * self.max_shadow_intensity * 0.8  # Very light
        
        # Subtle brow bone definition (much smaller and lighter)
        cv2.ellipse(brow_shadow_mask, 
                   (left_brow_center[0] - int(w*0.02), left_brow_center[1] - int(h*0.015)),
                   (max(4, int(w*0.015)), max(2, int(h*0.008))), 
                   0, 0, 360, shadow_intensity, -1)
        cv2.ellipse(brow_shadow_mask, 
                   (right_brow_center[0] + int(w*0.02), right_brow_center[1] - int(h*0.015)),
                   (max(4, int(w*0.015)), max(2, int(h*0.008))), 
                   0, 0, 360, shadow_intensity, -1)
        
        # NATURAL inner corner highlights (tear duct area)
        eye_highlight_mask = np.zeros((h, w), dtype=np.float32)
        
        inner_corner_intensity = intensity * self.max_highlight_intensity * 1.5  # Subtle glow
        
        left_inner_corner = (int(w * (0.5 - eye_spacing/2 + 0.02)), int(h * eye_y_ratio + int(h*0.01)))
        right_inner_corner = (int(w * (0.5 + eye_spacing/2 - 0.02)), int(h * eye_y_ratio + int(h*0.01)))
        
        cv2.circle(eye_highlight_mask, left_inner_corner, 
                  max(3, int(w*0.008)), inner_corner_intensity, -1)
        cv2.circle(eye_highlight_mask, right_inner_corner,
                  max(3, int(w*0.008)), inner_corner_intensity, -1)
        
        # Apply all enhancements with natural blending
        result = self._apply_highlighting_mask(result, brow_highlight_mask)
        result = self._apply_highlighting_mask(result, eye_highlight_mask)
        result = self._apply_darkening_mask(result, brow_shadow_mask)
        
        return result
    
    def _apply_nose_contouring(self, face: np.ndarray, landmarks: Optional[Any],
                              intensity: float) -> np.ndarray:
        """Apply NATURAL nose enhancement with organic shapes."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Adaptive nose positioning based on face proportions
        face_ratio = w / h
        if face_ratio > 0.9:  # Wide face
            nose_start_ratio = 0.38
            nose_end_ratio = 0.60
            nose_width_ratio = 0.012
        elif face_ratio < 0.75:  # Narrow face
            nose_start_ratio = 0.36
            nose_end_ratio = 0.62
            nose_width_ratio = 0.010
        else:  # Normal proportions
            nose_start_ratio = 0.37
            nose_end_ratio = 0.61
            nose_width_ratio = 0.011
        
        nose_center_x = int(w * 0.5)
        nose_start_y = int(h * nose_start_ratio)
        nose_end_y = int(h * nose_end_ratio)
        nose_width = max(3, int(w * nose_width_ratio))
        
        # NATURAL nose bridge shadows (soft curves, not harsh lines)
        nose_shadow_mask = np.zeros((h, w), dtype=np.float32)
        
        shadow_intensity = intensity * self.max_shadow_intensity * 1.2
        
        # Create natural curved nose shadows instead of rectangles
        shadow_points = []
        for y in range(nose_start_y, nose_end_y):
            progress = (y - nose_start_y) / (nose_end_y - nose_start_y)
            # Natural nose curve - wider at bridge, narrower towards tip
            current_width = nose_width * (1.2 - 0.4 * progress)
            
            # Left shadow curve
            left_x = int(nose_center_x - current_width)
            cv2.circle(nose_shadow_mask, (left_x, y), 
                      max(1, int(current_width * 0.3)), shadow_intensity * (1 - progress * 0.3), -1)
            
            # Right shadow curve
            right_x = int(nose_center_x + current_width)
            cv2.circle(nose_shadow_mask, (right_x, y), 
                      max(1, int(current_width * 0.3)), shadow_intensity * (1 - progress * 0.3), -1)
        
        # NATURAL nose bridge highlight (gentle ellipse, not rectangle)
        nose_highlight_mask = np.zeros((h, w), dtype=np.float32)
        
        highlight_intensity = intensity * self.max_highlight_intensity * 2.5
        
        # Main bridge highlight - natural elliptical shape
        bridge_center_y = int((nose_start_y + nose_end_y) * 0.5)
        bridge_length = nose_end_y - nose_start_y
        
        cv2.ellipse(nose_highlight_mask, (nose_center_x, bridge_center_y),
                   (max(2, int(nose_width * 0.4)), max(6, int(bridge_length * 0.35))),
                   0, 0, 360, highlight_intensity, -1)
        
        # NATURAL nose tip highlight (organic shape)
        tip_y = int(h * (nose_end_ratio - 0.02))
        tip_size = max(2, int(w * 0.008))
        
        cv2.circle(nose_highlight_mask, (nose_center_x, tip_y),
                  tip_size, highlight_intensity * 0.8, -1)
        
        # Apply natural nose contouring
        result = self._apply_darkening_mask(result, nose_shadow_mask)
        result = self._apply_highlighting_mask(result, nose_highlight_mask)
        
        return result
    
    def _apply_cheek_contouring(self, face: np.ndarray, landmarks: Optional[Any],
                               intensity: float) -> np.ndarray:
        """Apply NATURAL cheek enhancement with organic shapes."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Adaptive cheek positioning based on face proportions
        face_ratio = w / h
        if face_ratio > 0.9:  # Wide face
            cheek_y_start = 0.50
            cheek_y_end = 0.68
            cheek_width_ratio = 0.15
            highlight_y = 0.48
        elif face_ratio < 0.75:  # Narrow face
            cheek_y_start = 0.52
            cheek_y_end = 0.72
            cheek_width_ratio = 0.12
            highlight_y = 0.50
        else:  # Normal proportions
            cheek_y_start = 0.51
            cheek_y_end = 0.70
            cheek_width_ratio = 0.13
            highlight_y = 0.49
        
        # NATURAL cheek shadows (soft curves, not harsh polygons)
        cheek_shadow_mask = np.zeros((h, w), dtype=np.float32)
        shadow_intensity = intensity * self.max_shadow_intensity * 1.8
        
        # Create natural cheek hollow shadows with elliptical shapes
        left_cheek_center = (int(w * 0.22), int(h * (cheek_y_start + cheek_y_end) * 0.5))
        right_cheek_center = (int(w * 0.78), int(h * (cheek_y_start + cheek_y_end) * 0.5))
        
        cheek_width = max(12, int(w * cheek_width_ratio))
        cheek_height = max(8, int(h * (cheek_y_end - cheek_y_start) * 0.4))
        
        # Left cheek shadow - natural hollow shape
        cv2.ellipse(cheek_shadow_mask, left_cheek_center,
                   (cheek_width, cheek_height), -25, 0, 360, shadow_intensity, -1)
        
        # Right cheek shadow - natural hollow shape
        cv2.ellipse(cheek_shadow_mask, right_cheek_center,
                   (cheek_width, cheek_height), 25, 0, 360, shadow_intensity, -1)
        
        # NATURAL cheekbone highlights (subtle, organic)
        cheek_highlight_mask = np.zeros((h, w), dtype=np.float32)
        highlight_intensity = intensity * self.max_highlight_intensity * 3.0  # Still subtle
        
        left_highlight_center = (int(w * 0.26), int(h * highlight_y))
        right_highlight_center = (int(w * 0.74), int(h * highlight_y))
        
        highlight_width = max(8, int(w * 0.06))
        highlight_height = max(4, int(h * 0.02))
        
        # Natural cheekbone highlights following facial structure
        cv2.ellipse(cheek_highlight_mask, left_highlight_center,
                   (highlight_width, highlight_height), -20, 0, 360, highlight_intensity, -1)
        
        cv2.ellipse(cheek_highlight_mask, right_highlight_center,
                   (highlight_width, highlight_height), 20, 0, 360, highlight_intensity, -1)
        
        # Add subtle apple-of-cheek glow for natural warmth
        apple_highlight_mask = np.zeros((h, w), dtype=np.float32)
        apple_intensity = intensity * self.max_highlight_intensity * 1.5
        
        left_apple = (int(w * 0.30), int(h * (highlight_y + 0.03)))
        right_apple = (int(w * 0.70), int(h * (highlight_y + 0.03)))
        
        apple_size = max(6, int(w * 0.025))
        
        cv2.circle(apple_highlight_mask, left_apple, apple_size, apple_intensity, -1)
        cv2.circle(apple_highlight_mask, right_apple, apple_size, apple_intensity, -1)
        
        # Apply natural cheek enhancements
        result = self._apply_darkening_mask(result, cheek_shadow_mask)
        result = self._apply_highlighting_mask(result, cheek_highlight_mask)
        result = self._apply_highlighting_mask(result, apple_highlight_mask)
        
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
        """Extract or estimate lip coordinates with improved detection."""
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
        
        # ПОКРАЩЕНЕ ВИЗНАЧЕННЯ ПОЗИЦІЙ ГУБ на основі аналізу обличчя
        h, w = face_shape[:2]
        
        # Спробувати знайти губи за кольором
        detected_mouth = self._detect_mouth_by_color(self._current_face) if hasattr(self, '_current_face') else None
        
        if detected_mouth:
            return detected_mouth
        
        # Покращене оцінювання позицій з урахуванням пропорцій обличчя
        # Визначити тип обличчя для кращого позиціонування
        face_ratio = w / h
        
        if face_ratio > 0.9:  # Широке обличчя
            mouth_y_ratio = 0.72
            mouth_width_ratio = 0.18
        elif face_ratio < 0.75:  # Вузьке обличчя  
            mouth_y_ratio = 0.75
            mouth_width_ratio = 0.15
        else:  # Звичайне обличчя
            mouth_y_ratio = 0.74
            mouth_width_ratio = 0.16
        
        return {
            'upper_lip_center': [int(w*0.5), int(h*mouth_y_ratio)],
            'lower_lip_center': [int(w*0.5), int(h*(mouth_y_ratio + 0.04))],
            'left_corner': [int(w*(0.5 - mouth_width_ratio/2)), int(h*(mouth_y_ratio + 0.02))],
            'right_corner': [int(w*(0.5 + mouth_width_ratio/2)), int(h*(mouth_y_ratio + 0.02))],
            'mouth_center': [int(w*0.5), int(h*(mouth_y_ratio + 0.02))],
            'mouth_width': int(w*mouth_width_ratio),
            'mouth_height': int(h*0.04)
        }

    def _detect_mouth_by_color(self, face_region: np.ndarray) -> Optional[Dict]:
        """Advanced mouth detection by color analysis and edge detection."""
        try:
            h, w = face_region.shape[:2]
            
            # Method 1: Color-based detection (improved)
            face_hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            # Extended red ranges for lips
            lower_red1 = np.array([0, 30, 30])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 30, 30])  
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(face_hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(face_hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Find red contours
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mouth_candidates = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                lip_x, lip_y, lip_w, lip_h = cv2.boundingRect(contour)
                
                # More flexible size and position criteria
                if (area > 30 and lip_w > w * 0.05 and lip_h > h * 0.015 and 
                    lip_y > h * 0.55 and lip_y < h * 0.85 and 
                    lip_x > w * 0.15 and lip_x < w * 0.85):
                    
                    mouth_center = [lip_x + lip_w//2, lip_y + lip_h//2]
                    score = area * (1.0 - abs((mouth_center[1]) - h*0.72))
                    
                    mouth_candidates.append({
                        'center': mouth_center,
                        'bbox': (lip_x, lip_y, lip_w, lip_h),
                        'area': area,
                        'score': score
                    })
            
            # Method 2: Shadow detection fallback
            if not mouth_candidates:
                # Look for dark horizontal lines (mouth shadows)
                face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                
                # Focus on mouth area
                mouth_region_y1 = int(h * 0.65)
                mouth_region_y2 = int(h * 0.85)
                mouth_region_x1 = int(w * 0.25)
                mouth_region_x2 = int(w * 0.75)
                
                mouth_region = face_gray[mouth_region_y1:mouth_region_y2, mouth_region_x1:mouth_region_x2]
                
                if mouth_region.size > 0:
                    # Find darkest horizontal line (potential mouth line)
                    horizontal_means = np.mean(mouth_region, axis=1)
                    darkest_row = np.argmin(horizontal_means)
                    
                    actual_mouth_y = mouth_region_y1 + darkest_row
                    estimated_mouth_x = mouth_region_x1 + mouth_region.shape[1] // 2
                    
                    # Create candidate from detected shadow
                    estimated_width = int(w * 0.18)  # Reasonable mouth width
                    estimated_height = int(h * 0.03)  # Reasonable mouth height
                    
                    mouth_candidates.append({
                        'center': [estimated_mouth_x, actual_mouth_y],
                        'bbox': (estimated_mouth_x - estimated_width//2, actual_mouth_y - estimated_height//2,
                                estimated_width, estimated_height),
                        'area': estimated_width * estimated_height,
                        'score': 100  # Give decent score to shadow-based detection
                    })
            
            # Use best candidate
            if mouth_candidates:
                best_candidate = max(mouth_candidates, key=lambda x: x['score'])
                
                mouth_center = best_candidate['center']
                lip_x, lip_y, lip_w, lip_h = best_candidate['bbox']
                
                return {
                    'upper_lip_center': [mouth_center[0], lip_y],
                    'lower_lip_center': [mouth_center[0], lip_y + lip_h],
                    'left_corner': [lip_x, mouth_center[1]],
                    'right_corner': [lip_x + lip_w, mouth_center[1]],
                    'mouth_center': mouth_center,
                    'mouth_width': lip_w,
                    'mouth_height': lip_h
                }
                
        except Exception:
            pass
        
        return None
    
    def _apply_landmark_based_lip_contouring(self, face: np.ndarray, lip_coords: Dict, 
                                           intensity: float) -> np.ndarray:
        """Apply precise lip contouring using landmark coordinates."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Enhanced Cupid's bow with precise positioning
        lip_highlight_mask = np.zeros((h, w), dtype=np.float32)
        
        upper_lip = lip_coords['upper_lip_center']
        mouth_width = lip_coords['mouth_width']
        
        # ADAPTIVE Enhanced Cupid's bow - розміри залежать від справжнього розміру рота
        cupid_bow_center = upper_lip
        
        # Adaptive sizing based on detected mouth width
        if mouth_width < 30:  # Маленький рот
            cupid_radius = max(6, int(mouth_width * 0.2))
            peak_offset = max(8, int(mouth_width * 0.3))
            peak_radius = max(4, int(mouth_width * 0.15))
        elif mouth_width > 60:  # Великий рот
            cupid_radius = max(8, int(mouth_width * 0.12))
            peak_offset = max(10, int(mouth_width * 0.15))
            peak_radius = max(6, int(mouth_width * 0.1))
        else:  # Середній рот
            cupid_radius = max(7, int(mouth_width * 0.15))
            peak_offset = max(9, int(mouth_width * 0.2))
            peak_radius = max(5, int(mouth_width * 0.12))
        
        # Central dip (адаптивний розмір)
        cv2.circle(lip_highlight_mask, tuple(cupid_bow_center),
                  cupid_radius, intensity * 0.7, -1)  # Підвищена інтенсивність
        
        # Side peaks (адаптивні)
        left_peak = [cupid_bow_center[0] - peak_offset, cupid_bow_center[1] - 3]
        right_peak = [cupid_bow_center[0] + peak_offset, cupid_bow_center[1] - 3]
        
        cv2.circle(lip_highlight_mask, tuple(left_peak),
                  peak_radius, intensity * 0.6, -1)
        cv2.circle(lip_highlight_mask, tuple(right_peak),
                  peak_radius, intensity * 0.6, -1)
        
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
        """Add enhanced perioral area enhancements with vertical lip lines and marionette lines."""
        if not lip_coords:
            return face
            
        h, w = face.shape[:2]
        result = face.copy()
        
        # Enhanced philtrum area (area between nose and upper lip)
        philtrum_mask = np.zeros((h, w), dtype=np.float32)
        
        upper_lip = lip_coords['upper_lip_center']
        philtrum_center = [upper_lip[0], upper_lip[1] - max(8, int(lip_coords['mouth_height'] * 1.5))]  # Було 5, 1.2
        
        # More prominent philtrum lines
        cv2.line(philtrum_mask, 
                (philtrum_center[0] - 2, philtrum_center[1]), 
                (upper_lip[0] - 3, upper_lip[1]), 
                intensity * 0.2, 2)  # Було 0.1, товщина 1
        cv2.line(philtrum_mask, 
                (philtrum_center[0] + 2, philtrum_center[1]), 
                (upper_lip[0] + 3, upper_lip[1]), 
                intensity * 0.2, 2)  # Було 0.1, товщина 1
        
        # ДОДАТИ ВЕРТИКАЛЬНІ ЛІНІЇ НАД ГУБАМИ (8-10 ліній)
        perioral_lines_mask = np.zeros((h, w), dtype=np.float32)
        
        mouth_width = lip_coords['mouth_width']
        left_corner = lip_coords['left_corner']
        right_corner = lip_coords['right_corner']
        
        # ADAPTIVE perioral lines based on actual mouth width
        line_count = max(6, min(12, mouth_width // 4))  # Адаптивна кількість ліній
        
        for i in range(line_count):
            t = i / (line_count - 1) if line_count > 1 else 0.5
            x_pos = int(left_corner[0] + t * (right_corner[0] - left_corner[0]))
            
            # Adaptive line length based on mouth height
            line_length = max(8, int(mouth_width * 0.12))  # Довші лінії для великих ротів
            start_y = upper_lip[1] - line_length
            end_y = upper_lip[1] - 3
            
            # Stronger lines for better visibility
            cv2.line(perioral_lines_mask, (x_pos, start_y), (x_pos, end_y), 
                    intensity * 0.35, 3)  # Підвищена інтенсивність та товщина
        
        # ADAPTIVE MARIONETTE LINES (розмір залежить від рота)
        marionette_mask = np.zeros((h, w), dtype=np.float32)
        
        # Adaptive marionette length based on face proportions
        mouth_height = lip_coords.get('mouth_height', 20)
        marionette_length = max(12, min(30, int(mouth_height * 2.5)))  # Пропорційно до висоти рота
        
        # Left marionette line (адаптивний)
        left_start = left_corner
        left_end = [left_corner[0], left_corner[1] + marionette_length]
        
        cv2.line(marionette_mask, tuple(left_start), tuple(left_end), 
                intensity * 0.4, 4)  # Підвищена інтенсивність та товщина
        
        # Right marionette line (адаптивний)
        right_start = right_corner
        right_end = [right_corner[0], right_corner[1] + marionette_length]
        
        cv2.line(marionette_mask, tuple(right_start), tuple(right_end), 
                intensity * 0.4, 4)  # Підвищена інтенсивність та товщина
        
        # Enhanced corner shadows for definition
        corner_enhance_mask = np.zeros((h, w), dtype=np.float32)
        
        # Larger corner shadows
        corner_size = max(3, int(lip_coords['mouth_width'] * 0.08))  # Було 1, 0.05
        cv2.circle(corner_enhance_mask, tuple(left_corner), corner_size, intensity * 0.2, -1)  # Було 0.12
        cv2.circle(corner_enhance_mask, tuple(right_corner), corner_size, intensity * 0.2, -1)  # Було 0.12
        
        # Apply all perioral enhancements
        result = self._apply_darkening_mask(result, philtrum_mask)
        result = self._apply_darkening_mask(result, perioral_lines_mask)
        result = self._apply_darkening_mask(result, marionette_mask)
        result = self._apply_darkening_mask(result, corner_enhance_mask)
        
        return result
    
    def _apply_chin_jaw_contouring(self, face: np.ndarray, landmarks: Optional[Any],
                                  intensity: float) -> np.ndarray:
        """Apply NATURAL chin and jawline enhancement."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Adaptive positioning based on face proportions
        face_ratio = w / h
        if face_ratio > 0.9:  # Wide face
            chin_y = 0.92
            jaw_start_y = 0.78
            jaw_curve_intensity = 1.2
        elif face_ratio < 0.75:  # Narrow face
            chin_y = 0.90
            jaw_start_y = 0.76
            jaw_curve_intensity = 0.8
        else:  # Normal proportions
            chin_y = 0.91
            jaw_start_y = 0.77
            jaw_curve_intensity = 1.0
        
        # NATURAL under-chin shadow (soft, not harsh)
        chin_shadow_mask = np.zeros((h, w), dtype=np.float32)
        shadow_intensity = intensity * self.max_shadow_intensity * 1.5
        
        chin_shadow_center = (int(w * 0.5), int(h * (chin_y + 0.04)))
        chin_shadow_width = max(8, int(w * 0.08))
        chin_shadow_height = max(4, int(h * 0.04))
        
        cv2.ellipse(chin_shadow_mask, chin_shadow_center,
                   (chin_shadow_width, chin_shadow_height), 
                   0, 0, 360, shadow_intensity, -1)
        
        # NATURAL jawline enhancement (subtle curves, not polygons)
        jaw_shadow_mask = np.zeros((h, w), dtype=np.float32)
        jaw_intensity = intensity * self.max_shadow_intensity * jaw_curve_intensity
        
        # Create natural jaw curves using connected ellipses
        jaw_points_y = int(h * jaw_start_y)
        jaw_end_y = int(h * (chin_y - 0.02))
        
        # Left jawline curve
        for i, y_progress in enumerate(np.linspace(0, 1, 5)):
            y_pos = int(jaw_points_y + y_progress * (jaw_end_y - jaw_points_y))
            x_pos = int(w * (0.08 + y_progress * 0.32))  # Natural jaw curve
            
            curve_intensity = jaw_intensity * (1 - y_progress * 0.3)  # Fade towards chin
            curve_size = max(3, int(w * (0.015 - y_progress * 0.005)))
            
            cv2.circle(jaw_shadow_mask, (x_pos, y_pos), 
                      curve_size, curve_intensity, -1)
        
        # Right jawline curve
        for i, y_progress in enumerate(np.linspace(0, 1, 5)):
            y_pos = int(jaw_points_y + y_progress * (jaw_end_y - jaw_points_y))
            x_pos = int(w * (0.92 - y_progress * 0.32))  # Natural jaw curve
            
            curve_intensity = jaw_intensity * (1 - y_progress * 0.3)  # Fade towards chin
            curve_size = max(3, int(w * (0.015 - y_progress * 0.005)))
            
            cv2.circle(jaw_shadow_mask, (x_pos, y_pos), 
                      curve_size, curve_intensity, -1)
        
        # NATURAL chin highlight (subtle, organic)
        chin_highlight_mask = np.zeros((h, w), dtype=np.float32)
        highlight_intensity = intensity * self.max_highlight_intensity * 2.0
        
        chin_highlight_center = (int(w * 0.5), int(h * chin_y))
        highlight_size = max(4, int(w * 0.012))
        
        cv2.circle(chin_highlight_mask, chin_highlight_center,
                  highlight_size, highlight_intensity, -1)
        
        # Apply natural chin and jaw enhancements
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
        """Advanced skin tone analysis for natural makeup color selection."""
        # Multiple color space analysis for comprehensive skin tone understanding
        lab_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB)
        hsv_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)
        
        # Focus on central skin areas (avoid hair, background, shadows)
        h, w = image.shape[:2]
        central_region = image[int(h*0.3):int(h*0.7), int(w*0.25):int(w*0.75)]
        central_lab = lab_image[int(h*0.3):int(h*0.7), int(w*0.25):int(w*0.75)]
        central_hsv = hsv_image[int(h*0.3):int(h*0.7), int(w*0.25):int(w*0.75)]
        
        # Advanced skin tone metrics
        l_mean = np.mean(central_lab[:, :, 0])  # Lightness
        a_mean = np.mean(central_lab[:, :, 1])  # Green-Red axis  
        b_mean = np.mean(central_lab[:, :, 2])  # Blue-Yellow axis
        
        # HSV analysis for better undertone detection
        h_mean = np.mean(central_hsv[:, :, 0])  # Hue
        s_mean = np.mean(central_hsv[:, :, 1])  # Saturation
        v_mean = np.mean(central_hsv[:, :, 2])  # Value/Brightness
        
        # RGB analysis for natural color matching
        r_mean = np.mean(central_region[:, :, 2])  # Red channel
        g_mean = np.mean(central_region[:, :, 1])  # Green channel
        b_channel_mean = np.mean(central_region[:, :, 0])  # Blue channel
        
        # Sophisticated undertone detection
        warmth_score = (b_mean - 128) + (h_mean < 30 or h_mean > 300) * 20
        
        # Detailed skin classification
        is_very_light = l_mean > 180
        is_light = l_mean > 150 and l_mean <= 180  
        is_medium = l_mean > 120 and l_mean <= 150
        is_dark = l_mean > 90 and l_mean <= 120
        is_very_dark = l_mean <= 90
        
        is_warm = warmth_score > 5
        is_neutral = abs(warmth_score) <= 5
        is_cool = warmth_score < -5
        
        return {
            'lightness': l_mean,
            'warmth': warmth_score,
            'saturation': s_mean,
            'hue': h_mean,
            'brightness': v_mean,
            'rgb': [r_mean, g_mean, b_channel_mean],
            'lab': [l_mean, a_mean, b_mean],
            'is_very_light': is_very_light,
            'is_light': is_light, 
            'is_medium': is_medium,
            'is_dark': is_dark,
            'is_very_dark': is_very_dark,
            'is_warm': is_warm,
            'is_neutral': is_neutral,
            'is_cool': is_cool,
            'undertone_strength': abs(warmth_score)
        }
    
    def _create_gradient_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create natural gradient mask for seamless blending."""
        gradient_mask = mask.copy().astype(np.float32)
        
        # NATURAL blending - multiple blur levels for seamless transitions
        blur_sizes = [3, 7, 15]  # Larger blur sizes for natural falloff
        weights = [0.2, 0.5, 0.3]  # More weight on medium blur for natural look
        
        blended_mask = np.zeros_like(gradient_mask)
        
        for blur_size, weight in zip(blur_sizes, weights):
            blurred = cv2.GaussianBlur(gradient_mask, (blur_size, blur_size), 0)
            blended_mask += blurred * weight
        
        # Normalize and apply gentle power curve for natural falloff
        blended_mask = blended_mask / np.sum(weights)
        blended_mask = np.power(blended_mask, 0.8)  # Gentle curve for smooth transitions
        
        # Additional smoothing for ultra-natural blending
        blended_mask = cv2.GaussianBlur(blended_mask, (5, 5), 0)
        
        return blended_mask
    
    def _get_adaptive_darkening_factor(self, skin_tone: Dict[str, float]) -> float:
        """Calculate NATURAL darkening factor for subtle contouring."""
        # NATURAL base factors - much more subtle
        if skin_tone['is_very_light']:
            base_factor = 0.95   # Very subtle for very light skin
        elif skin_tone['is_light']:
            base_factor = 0.90   # Light skin needs gentle shadowing
        elif skin_tone['is_medium']:
            base_factor = 0.85   # Medium skin can handle slightly more
        elif skin_tone['is_dark']:
            base_factor = 0.80   # Dark skin needs more contrast
        else:  # very_dark
            base_factor = 0.75   # Very dark skin needs most contrast
        
        # Fine-tune based on undertones (very subtle adjustments)
        if skin_tone['is_warm']:
            base_factor *= 0.98   # Warm tones slightly less darkening
        elif skin_tone['is_cool']:
            base_factor *= 0.96   # Cool tones slightly more darkening
        
        # Ensure we never go below natural limits
        return max(0.70, min(0.98, base_factor))
    
    def _get_adaptive_highlighting_factor(self, skin_tone: Dict[str, float]) -> float:
        """Calculate NATURAL highlighting factor for subtle enhancement."""
        # NATURAL base factors - subtle enhancement only
        if skin_tone['is_very_light']:
            base_factor = 1.08   # Very subtle for very light skin
        elif skin_tone['is_light']:
            base_factor = 1.12   # Light skin gentle highlighting
        elif skin_tone['is_medium']:
            base_factor = 1.18   # Medium skin moderate highlighting
        elif skin_tone['is_dark']:
            base_factor = 1.25   # Dark skin more visible highlighting
        else:  # very_dark
            base_factor = 1.35   # Very dark skin needs most highlighting
        
        # Fine-tune based on undertones (very subtle adjustments)
        if skin_tone['is_warm']:
            base_factor *= 1.05   # Warm tones slightly more highlighting
        elif skin_tone['is_cool']:
            base_factor *= 1.03   # Cool tones slightly less highlighting
        
        # Ensure we stay within natural limits
        return max(1.05, min(1.40, base_factor))
    
    def _create_shadow_color(self, channel: np.ndarray, skin_tone: Dict[str, float], 
                           color_channel: int) -> np.ndarray:
        """Create NATURAL shadow color that harmonizes with existing skin tone."""
        # Use the natural darkening factor (0.70-0.98)
        darkening_factor = self._get_adaptive_darkening_factor(skin_tone)
        shadow_base = channel * darkening_factor
        
        # Get the person's natural skin RGB values for color matching
        skin_rgb = skin_tone['rgb']
        current_channel_value = skin_rgb[2-color_channel]  # Convert BGR to RGB indexing
        
        # Create shadows that are natural variations of existing skin color
        if color_channel == 0:  # Blue channel
            if skin_tone['is_warm']:
                # Warm skin: reduce blue for natural warm shadows
                adjustment = 0.95 + (current_channel_value / 255) * 0.03  # Very subtle
            else:
                # Cool skin: maintain blue for natural cool shadows  
                adjustment = 0.98 + (current_channel_value / 255) * 0.02
        elif color_channel == 1:  # Green channel
            # Reduce green slightly for all skin tones (natural shadow characteristic)
            adjustment = 0.94 + (current_channel_value / 255) * 0.04
        elif color_channel == 2:  # Red channel
            if skin_tone['is_warm']:
                # Warm skin: maintain red warmth in shadows
                adjustment = 0.98 + (current_channel_value / 255) * 0.02
            else:
                # Cool skin: reduce red slightly for cooler shadows
                adjustment = 0.96 + (current_channel_value / 255) * 0.03
        
        # Apply natural adjustment (very subtle color shifts)
        shadow_base = shadow_base * adjustment
        
        return shadow_base
    
    def _create_highlight_color(self, channel: np.ndarray, skin_tone: Dict[str, float], 
                              color_channel: int) -> np.ndarray:
        """Create NATURAL highlight color that enhances existing skin tone."""
        # Use the natural highlighting factor (1.05-1.40)
        highlighting_factor = self._get_adaptive_highlighting_factor(skin_tone)
        highlight_base = np.minimum(channel * highlighting_factor, 255)
        
        # Get the person's natural skin RGB values for color matching
        skin_rgb = skin_tone['rgb']
        current_channel_value = skin_rgb[2-color_channel]  # Convert BGR to RGB indexing
        
        # Create highlights that naturally enhance existing skin tone
        if color_channel == 0:  # Blue channel
            if skin_tone['is_warm']:
                # Warm skin: slightly reduce blue for warm highlights
                adjustment = 0.97 + (current_channel_value / 255) * 0.02  # Very subtle
            else:
                # Cool skin: maintain blue for natural cool highlights
                adjustment = 0.99 + (current_channel_value / 255) * 0.01
        elif color_channel == 1:  # Green channel
            if skin_tone['is_warm']:
                # Warm skin: enhance green for natural warm glow
                adjustment = 1.02 + (current_channel_value / 255) * 0.01
            else:
                # Cool skin: slightly enhance green for natural radiance
                adjustment = 1.01 + (current_channel_value / 255) * 0.01
        elif color_channel == 2:  # Red channel
            if skin_tone['is_warm']:
                # Warm skin: enhance red for natural warm glow
                adjustment = 1.03 + (current_channel_value / 255) * 0.02
            else:
                # Cool skin: slight red enhancement for healthy look
                adjustment = 1.01 + (current_channel_value / 255) * 0.01
        
        # Apply natural adjustment (very subtle color enhancement)
        highlight_base = np.minimum(highlight_base * adjustment, 255)
        
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
            
    def _validate_input_image(self, image: np.ndarray) -> bool:
        """Validate input image quality and properties."""
        if image is None or image.size == 0:
            return False
            
        h, w = image.shape[:2]
        if h < 100 or w < 100:  # Too small for meaningful enhancement
            return False
            
        if len(image.shape) != 3:  # Must be color image
            return False
            
        # Check for reasonable dynamic range
        if np.std(image) < 10:  # Too flat/uniform
            return False
            
        return True
    
    def _validate_face_region(self, face_region: np.ndarray) -> bool:
        """Validate face region quality for safe enhancement."""
        if face_region.size == 0:
            return False
            
        h, w = face_region.shape[:2]
        if h < 64 or w < 64:  # Too small for detailed enhancement
            return False
            
        # Check if region contains meaningful facial features
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Basic feature detection - look for variation that suggests facial features
        if np.std(gray) < 15:  # Too uniform to be a real face
            return False
            
        return True
    
    def _apply_safety_limits(self, base_intensity: float, face_region: np.ndarray) -> float:
        """Apply safety limits to prevent unnatural enhancement."""
        # Analyze face region to determine safe intensity limits
        skin_tone = self._analyze_skin_tone(face_region)
        
        # Calculate maximum safe intensity based on skin characteristics
        if skin_tone['is_very_light']:
            max_safe_intensity = 0.08  # Very conservative for light skin
        elif skin_tone['is_light']:
            max_safe_intensity = 0.12  # Conservative for light skin
        elif skin_tone['is_medium']:
            max_safe_intensity = 0.15  # Moderate for medium skin
        else:
            max_safe_intensity = 0.18  # Slightly higher for darker skin
        
        # Additional safety based on image quality
        brightness = np.mean(face_region)
        if brightness < 80:  # Very dark image
            max_safe_intensity *= 0.8
        elif brightness > 200:  # Very bright image
            max_safe_intensity *= 0.7
        
        return min(base_intensity, max_safe_intensity)
    
    def _validate_step_result(self, result: np.ndarray, original: np.ndarray, step_name: str) -> np.ndarray:
        """Validate each contouring step result for natural appearance."""
        # Calculate difference between result and original
        diff = np.abs(result.astype(np.float32) - original.astype(np.float32))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Safety thresholds for natural appearance
        MAX_PIXEL_CHANGE = 30   # No pixel should change more than this
        MAX_MEAN_CHANGE = 8     # Overall change should be subtle
        
        if max_diff > MAX_PIXEL_CHANGE or mean_diff > MAX_MEAN_CHANGE:
            print(f"Warning: {step_name} step too intense (max: {max_diff:.1f}, mean: {mean_diff:.1f}) - using original")
            return original  # Return original if change is too dramatic
        
        return result
    
    def _validate_final_result(self, final: np.ndarray, original: np.ndarray) -> bool:
        """Validate final result for overall natural appearance."""
        # Overall validation of final result
        diff = np.abs(final.astype(np.float32) - original.astype(np.float32))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Final safety thresholds
        FINAL_MAX_PIXEL_CHANGE = 25  # Maximum change for any pixel
        FINAL_MAX_MEAN_CHANGE = 6    # Overall change should be very subtle
        
        if max_diff > FINAL_MAX_PIXEL_CHANGE:
            print(f"Final validation failed: max pixel change too high ({max_diff:.1f})")
            return False
        
        if mean_diff > FINAL_MAX_MEAN_CHANGE:
            print(f"Final validation failed: mean change too high ({mean_diff:.1f})")
            return False
        
        return True
    
    def _blend_face_naturally(self, original_image: np.ndarray, enhanced_face: np.ndarray, 
                             bbox: List[int]) -> np.ndarray:
        """Blend enhanced face naturally back into original image."""
        result = original_image.copy()
        x1, y1, x2, y2 = bbox
        
        # Create natural blending mask with feathered edges
        h, w = enhanced_face.shape[:2]
        blend_mask = np.ones((h, w), dtype=np.float32)
        
        # Feather edges for seamless blending
        feather_size = min(h, w) // 10  # 10% feathering
        
        for i in range(feather_size):
            alpha = (i + 1) / feather_size
            # Top edge
            blend_mask[i, :] *= alpha
            # Bottom edge
            blend_mask[h-1-i, :] *= alpha
            # Left edge
            blend_mask[:, i] *= alpha
            # Right edge
            blend_mask[:, w-1-i] *= alpha
        
        # Apply natural blending
        for c in range(3):
            enhanced_channel = enhanced_face[:, :, c].astype(np.float32)
            original_channel = result[y1:y2, x1:x2, c].astype(np.float32)
            
            blended = blend_mask * enhanced_channel + (1 - blend_mask) * original_channel
            result[y1:y2, x1:x2, c] = np.clip(blended, 0, 255).astype(np.uint8)
        
        return result