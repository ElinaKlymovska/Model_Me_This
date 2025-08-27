"""Face enhancement and processing."""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional
from .facial_contouring import ContouringMask
from .expression_lines import ExpressionProcessor, ExpressionType


class FaceInpainting:
    """Advanced face enhancement using OpenCV techniques."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enhancement_strength = config.get('enhancement_strength', 1.5)
        self.sharpening_strength = config.get('sharpening_strength', 0.3)
        self.denoising_strength = config.get('denoising_strength', 10)
        
        # Initialize contouring and expression systems
        self.contouring_enabled = config.get('facial_enhancement', {}).get('contouring', {}).get('enabled', True)
        self.expression_lines_enabled = config.get('facial_enhancement', {}).get('expression_lines', {}).get('enabled', True)
        
        if self.contouring_enabled:
            self.contouring_mask = ContouringMask(config.get('facial_enhancement', {}))
        
        if self.expression_lines_enabled:
            self.expression_processor = ExpressionProcessor(config.get('facial_enhancement', {}))
        
    def process_face(self, image: np.ndarray, face_bbox: list, image_path: Optional[str] = None, 
                    landmarks: Optional[Dict] = None) -> np.ndarray:
        """Enhanced face processing with multiple techniques including contouring."""
        x1, y1, x2, y2 = face_bbox
        
        # Extract face region
        face_region = image[y1:y2, x1:x2].copy()
        if face_region.size == 0:
            return image
            
        # Apply basic enhancement techniques
        enhanced_face = self._enhance_face(face_region)
        
        # Apply contouring if enabled
        if self.contouring_enabled:
            enhanced_face = self.contouring_mask.apply_contouring(
                enhanced_face, [0, 0, enhanced_face.shape[1], enhanced_face.shape[0]], 
                image_path, landmarks
            )
        
        # Apply expression lines if enabled
        if self.expression_lines_enabled:
            # Auto-detect expression or default to neutral
            expression_type = self.expression_processor.detect_expression_type(enhanced_face)
            enhanced_face = self.expression_processor.apply_expression_lines(
                enhanced_face, [0, 0, enhanced_face.shape[1], enhanced_face.shape[0]], 
                expression_type, image_path
            )
        
        # Create smooth blending mask
        mask = self._create_blend_mask(face_region.shape[:2])
        
        # Blend enhanced face back to original image
        result = image.copy()
        
        # Resize enhanced face if needed
        if enhanced_face.shape[:2] != (y2-y1, x2-x1):
            enhanced_face = cv2.resize(enhanced_face, (x2-x1, y2-y1))
        
        # Apply blending
        for c in range(3):  # For each color channel
            face_channel = enhanced_face[:, :, c].astype(np.float32)
            orig_channel = result[y1:y2, x1:x2, c].astype(np.float32)
            
            blended = mask * face_channel + (1 - mask) * orig_channel
            result[y1:y2, x1:x2, c] = np.clip(blended, 0, 255).astype(np.uint8)
        
        return result
    
    def _enhance_face(self, face: np.ndarray) -> np.ndarray:
        """Apply various enhancement techniques to face."""
        enhanced = face.copy()
        
        # 1. Denoising
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, self.denoising_strength, self.denoising_strength, 7, 21)
        
        # 2. Contrast and brightness enhancement
        enhanced = self._enhance_contrast_brightness(enhanced)
        
        # 3. Sharpening
        enhanced = self._apply_sharpening(enhanced)
        
        # 4. Skin smoothing (selective blur)
        enhanced = self._smooth_skin(enhanced)
        
        # 5. Color correction
        enhanced = self._color_correction(enhanced)
        
        return enhanced
    
    def _enhance_contrast_brightness(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast and brightness using CLAHE."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for sharpening."""
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Create unsharp mask
        sharpened = cv2.addWeighted(image, 1 + self.sharpening_strength, blurred, -self.sharpening_strength, 0)
        
        return sharpened
    
    def _smooth_skin(self, image: np.ndarray) -> np.ndarray:
        """Apply selective smoothing to skin areas."""
        # Create a bilateral filter for skin smoothing
        smoothed = cv2.bilateralFilter(image, 15, 80, 80)
        
        # Create mask for skin areas (simple color-based approach)
        skin_mask = self._create_skin_mask(image)
        
        # Blend smoothed and original based on skin mask
        result = image.copy().astype(np.float32)
        smoothed = smoothed.astype(np.float32)
        
        for c in range(3):
            result[:, :, c] = skin_mask * smoothed[:, :, c] + (1 - skin_mask) * result[:, :, c]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _create_skin_mask(self, image: np.ndarray) -> np.ndarray:
        """Create a simple skin tone mask."""
        # Convert to YCrCb color space for skin detection
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create mask
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Convert to float and apply Gaussian blur for soft edges
        skin_mask = skin_mask.astype(np.float32) / 255.0
        skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)
        
        return skin_mask
    
    def _color_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply color correction for better skin tones."""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Slight adjustment to A and B channels for better skin tones
        a = cv2.add(a, 2)  # Slightly warmer
        b = cv2.subtract(b, 2)  # Slightly less yellow
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return corrected
    
    def _create_blend_mask(self, face_shape: tuple) -> np.ndarray:
        """Create a smooth blending mask for face region."""
        h, w = face_shape
        
        # Create elliptical mask
        center = (w // 2, h // 2)
        axes = (w // 2 - 5, h // 2 - 5)
        
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        
        # Apply Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def process_multiple_faces(self, image: np.ndarray, faces: List[Dict[str, Any]], 
                             image_path: Optional[str] = None) -> np.ndarray:
        """Process multiple faces in an image."""
        result = image.copy()
        
        for face in faces:
            # Extract landmarks from face detection if available
            landmarks = face.get('landmarks', None)
            result = self.process_face(result, face['bbox'], image_path, landmarks)
            
        return result