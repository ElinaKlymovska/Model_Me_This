"""
A-Pass pipeline integration for face enhancement.
Integrated with existing portrait-enhancer pipeline.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import cv2
import os
import sys

# Add portrait-enhancer to path if available
portrait_enhancer_path = Path(__file__).parent.parent.parent.parent / "portrait-enhancer"
if portrait_enhancer_path.exists():
    sys.path.insert(0, str(portrait_enhancer_path))

from ..utils.timing import Timer
from ..utils.io import load_image, save_image

try:
    from run_a_pass import create_face_mask, create_line_mask, create_ellipse_stroke_mask, merge_masks
    PORTRAIT_ENHANCER_AVAILABLE = True
except ImportError:
    PORTRAIT_ENHANCER_AVAILABLE = False
    print("Warning: portrait-enhancer not available, using fallback A-Pass")


class APassPipeline:
    """A-Pass pipeline for face enhancement.
    Integrates with existing portrait-enhancer pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize A-Pass pipeline.
        
        Args:
            config: A-Pass configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.model_path = config.get('model_path')
        self.batch_size = config.get('batch_size', 4)
        self.device = config.get('device', 'auto')
        self.workdir = config.get('workdir', 'work')
        self.model = None
        self._initialized = False
        
        if self.enabled:
            self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup A-Pass pipeline components."""
        if not self.enabled:
            return
        
        # Create work directories
        os.makedirs(os.path.join(self.workdir, "a_pass"), exist_ok=True)
        os.makedirs(os.path.join(self.workdir, "masks"), exist_ok=True)
        
        self._initialized = True
        
        if PORTRAIT_ENHANCER_AVAILABLE:
            print("✓ A-Pass: portrait-enhancer integration available")
        else:
            print("⚠ A-Pass: using fallback implementation")
    
    def process_faces(self, face_patches: List[np.ndarray], 
                     original_image: np.ndarray = None) -> List[np.ndarray]:
        """Process face patches through A-Pass.
        
        Args:
            face_patches: List of face patch images
            original_image: Original full image for context
            
        Returns:
            List of enhanced face patches
        """
        if not self.enabled or not self._initialized:
            return face_patches
        
        if not face_patches:
            return []
        
        with Timer("a_pass_processing") as timer:
            if PORTRAIT_ENHANCER_AVAILABLE and original_image is not None:
                # Use existing portrait-enhancer pipeline
                enhanced_patches = self._process_with_portrait_enhancer(original_image)
            else:
                # Use fallback enhancement
                enhanced_patches = self._process_fallback(face_patches)
            
            # Add timing information
            for patch in enhanced_patches:
                patch.metadata = getattr(patch, 'metadata', {})
                patch.metadata['a_pass_time'] = timer.elapsed_time
            
            return enhanced_patches
    
    def _process_with_portrait_enhancer(self, original_image: np.ndarray) -> List[np.ndarray]:
        """Process using existing portrait-enhancer pipeline.
        
        Args:
            original_image: Original full image
            
        Returns:
            Enhanced face patches
        """
        try:
            # Convert numpy array to PIL Image
            from PIL import Image
            pil_image = Image.fromarray(original_image)
            
            # Save original image temporarily
            temp_input = os.path.join(self.workdir, "temp_input.png")
            pil_image.save(temp_input)
            
            # Run A-Pass processing
            self._run_a_pass_processing(temp_input)
            
            # Load results
            base_enhanced_path = os.path.join(self.workdir, "a_pass", "base_enhanced.png")
            face_mask_path = os.path.join(self.workdir, "masks", "face_mask.png")
            
            if os.path.exists(base_enhanced_path):
                enhanced_image = np.array(Image.open(base_enhanced_path))
                face_mask = np.array(Image.open(face_mask_path))
                
                # Extract face patches using the mask
                enhanced_patches = self._extract_patches_with_mask(enhanced_image, face_mask)
                return enhanced_patches
            
        except Exception as e:
            print(f"Warning: portrait-enhancer processing failed: {e}")
            print("Falling back to basic enhancement")
        
        # Fallback to basic processing
        return self._process_fallback([])
    
    def _run_a_pass_processing(self, input_path: str):
        """Run the existing A-Pass processing.
        
        Args:
            input_path: Path to input image
        """
        try:
            # Import and run the existing A-Pass code
            from run_a_pass import main as run_a_pass_main
            
            # Temporarily modify sys.argv to simulate command line arguments
            import sys
            original_argv = sys.argv.copy()
            
            sys.argv = [
                'run_a_pass.py',
                '--input', input_path,
                '--workdir', self.workdir
            ]
            
            # Run A-Pass
            run_a_pass_main()
            
            # Restore original argv
            sys.argv = original_argv
            
        except Exception as e:
            print(f"Error running A-Pass: {e}")
            raise
    
    def _extract_patches_with_mask(self, enhanced_image: np.ndarray, 
                                  face_mask: np.ndarray) -> List[np.ndarray]:
        """Extract face patches using the generated mask.
        
        Args:
            enhanced_image: Enhanced image
            face_mask: Face mask
            
        Returns:
            List of enhanced face patches
        """
        # Find connected components in mask
        from scipy import ndimage
        labeled_mask, num_features = ndimage.label(face_mask > 128)
        
        patches = []
        for i in range(1, num_features + 1):
            # Get bounding box for this component
            coords = np.where(labeled_mask == i)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                # Extract patch with padding
                padding = 20
                y_min = max(0, y_min - padding)
                y_max = min(enhanced_image.shape[0], y_max + padding)
                x_min = max(0, x_min - padding)
                x_max = min(enhanced_image.shape[1], x_max + padding)
                
                patch = enhanced_image[y_min:y_max, x_min:x_max]
                patches.append(patch)
        
        return patches if patches else [enhanced_image]
    
    def _process_fallback(self, face_patches: List[np.ndarray]) -> List[np.ndarray]:
        """Fallback processing when portrait-enhancer is not available.
        
        Args:
            face_patches: List of face patch images
            
        Returns:
            Enhanced face patches
        """
        enhanced_patches = []
        
        for patch in face_patches:
            # Apply basic enhancement (placeholder)
            enhanced = self._apply_basic_enhancement(patch)
            enhanced_patches.append(enhanced)
        
        return enhanced_patches
    
    def _apply_basic_enhancement(self, face_patch: np.ndarray) -> np.ndarray:
        """Apply basic enhancement to face patch.
        
        Args:
            face_patch: Input face patch
            
        Returns:
            Enhanced face patch
        """
        # Placeholder enhancement - adjust brightness and contrast
        enhanced = face_patch.copy()
        
        # Convert to float for processing
        if enhanced.dtype == np.uint8:
            enhanced = enhanced.astype(np.float32) / 255.0
        
        # Basic enhancement
        enhanced = np.clip(enhanced * 1.1, 0, 1)  # Increase brightness
        
        # Convert back to uint8
        if face_patch.dtype == np.uint8:
            enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
    
    def integrate_with_detection(self, image: np.ndarray, 
                               detections: List[Dict[str, Any]]) -> np.ndarray:
        """Integrate A-Pass results back into the original image.
        
        Args:
            image: Original image
            detections: Face detections with enhanced patches
            
        Returns:
            Image with integrated A-Pass results
        """
        if not self.enabled or not self._initialized:
            return image
        
        # Process the entire image through A-Pass
        enhanced_patches = self.process_faces([], image)
        
        if enhanced_patches and len(enhanced_patches) > 0:
            # Use the first enhanced patch (should be the full enhanced image)
            return enhanced_patches[0]
        
        return image
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the A-Pass pipeline.
        
        Returns:
            Pipeline information
        """
        return {
            'enabled': self.enabled,
            'initialized': self._initialized,
            'model_path': str(self.model_path) if self.model_path else None,
            'batch_size': self.batch_size,
            'device': self.device,
            'workdir': self.workdir,
            'portrait_enhancer_available': PORTRAIT_ENHANCER_AVAILABLE
        }
    
    def is_available(self) -> bool:
        """Check if A-Pass pipeline is available.
        
        Returns:
            True if available, False otherwise
        """
        return self.enabled and self._initialized
