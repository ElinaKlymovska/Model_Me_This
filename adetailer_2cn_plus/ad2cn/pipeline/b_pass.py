"""
B-Pass pipeline integration for face enhancement.
Integrated with existing portrait-enhancer pipeline.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import cv2
import os
import sys
import yaml

# Add portrait-enhancer to path if available
portrait_enhancer_path = Path(__file__).parent.parent.parent.parent / "portrait-enhancer"
if portrait_enhancer_path.exists():
    sys.path.insert(0, str(portrait_enhancer_path))

from ..utils.timing import Timer
from ..utils.io import load_image, save_image

try:
    from run_b_pass import create_a1111_payload, run_a1111_inpainting
    PORTRAIT_ENHANCER_AVAILABLE = True
except ImportError:
    PORTRAIT_ENHANCER_AVAILABLE = False
    print("Warning: portrait-enhancer not available, using fallback B-Pass")


class BPassPipeline:
    """B-Pass pipeline for face enhancement.
    Integrates with existing portrait-enhancer pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize B-Pass pipeline.
        
        Args:
            config: B-Pass configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.model_path = config.get('model_path')
        self.batch_size = config.get('batch_size', 4)
        self.device = config.get('device', 'auto')
        self.workdir = config.get('workdir', 'work')
        self.output_dir = config.get('output_dir', 'output')
        self.config_path = config.get('config_path', 'config.yaml')
        self.model = None
        self._initialized = False
        
        if self.enabled:
            self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup B-Pass pipeline components."""
        if not self.enabled:
            return
        
        # Create work directories
        os.makedirs(self.workdir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._initialized = True
        
        if PORTRAIT_ENHANCER_AVAILABLE:
            print("✓ B-Pass: portrait-enhancer integration available")
        else:
            print("⚠ B-Pass: using fallback implementation")
    
    def process_faces(self, face_patches: List[np.ndarray], 
                     original_image: np.ndarray = None,
                     a_pass_results: List[np.ndarray] = None) -> List[np.ndarray]:
        """Process face patches through B-Pass.
        
        Args:
            face_patches: List of face patch images
            original_image: Original full image for context
            a_pass_results: Results from A-Pass processing
            
        Returns:
            List of enhanced face patches
        """
        if not self.enabled or not self._initialized:
            return face_patches
        
        if not face_patches and original_image is None:
            return []
        
        with Timer("b_pass_processing") as timer:
            if PORTRAIT_ENHANCER_AVAILABLE and original_image is not None:
                # Use existing portrait-enhancer pipeline
                enhanced_patches = self._process_with_portrait_enhancer(original_image, a_pass_results)
            else:
                # Use fallback enhancement
                enhanced_patches = self._process_fallback(face_patches)
            
            # Add timing information
            for patch in enhanced_patches:
                patch.metadata = getattr(patch, 'metadata', {})
                patch.metadata['b_pass_time'] = timer.elapsed_time
            
            return enhanced_patches
    
    def _process_with_portrait_enhancer(self, original_image: np.ndarray, 
                                      a_pass_results: List[np.ndarray] = None) -> List[np.ndarray]:
        """Process using existing portrait-enhancer pipeline.
        
        Args:
            original_image: Original full image
            a_pass_results: Results from A-Pass processing
            
        Returns:
            Enhanced face patches
        """
        try:
            # Check if A-Pass results exist
            a_pass_image_path = os.path.join(self.workdir, "a_pass", "base_enhanced.png")
            face_mask_path = os.path.join(self.workdir, "masks", "face_mask.png")
            contour_map_path = os.path.join(self.workdir, "a_pass", "contour_map.png")
            
            if not all(os.path.exists(p) for p in [a_pass_image_path, face_mask_path]):
                print("Warning: A-Pass results not found, running A-Pass first")
                # Run A-Pass if not already done
                self._run_a_pass_if_needed(original_image)
            
            # Run B-Pass processing
            output_name = "final_enhanced"
            output_path = os.path.join(self.output_dir, f"{output_name}.png")
            
            self._run_b_pass_processing(a_pass_image_path, face_mask_path, contour_map_path, output_path)
            
            # Load and return results
            if os.path.exists(output_path):
                enhanced_image = load_image(output_path)
                return [enhanced_image]
            
        except Exception as e:
            print(f"Warning: portrait-enhancer B-Pass processing failed: {e}")
            print("Falling back to basic enhancement")
        
        # Fallback to basic processing
        return self._process_fallback([])
    
    def _run_a_pass_if_needed(self, original_image: np.ndarray):
        """Run A-Pass if results don't exist.
        
        Args:
            original_image: Original image
        """
        try:
            # Import A-Pass pipeline
            from .a_pass import APassPipeline
            
            # Create temporary A-Pass config
            a_pass_config = {
                'enabled': True,
                'workdir': self.workdir
            }
            
            # Run A-Pass
            a_pass = APassPipeline(a_pass_config)
            a_pass.process_faces([], original_image)
            
        except Exception as e:
            print(f"Error running A-Pass: {e}")
            raise
    
    def _run_b_pass_processing(self, a_pass_image: str, face_mask_path: str, 
                              contour_map_path: str, output_path: str):
        """Run the existing B-Pass processing.
        
        Args:
            a_pass_image: Path to A-Pass enhanced image
            face_mask_path: Path to face mask
            contour_map_path: Path to contour map
            output_path: Output path for final result
        """
        try:
            # Load configuration
            config = self._load_config()
            
            # Run B-Pass
            run_a1111_inpainting(config, a_pass_image, face_mask_path, contour_map_path, output_path)
            
        except Exception as e:
            print(f"Error running B-Pass: {e}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration for B-Pass.
        
        Returns:
            Configuration dictionary
        """
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            # Create default config
            default_config = {
                "general": {
                    "a1111_endpoint": "http://127.0.0.1:7860",
                    "backend": "a1111"
                },
                "b_pass": {
                    "prompt": "portrait, high quality, detailed face, professional photography",
                    "negative": "blurry, low quality, distorted, artifacts",
                    "denoise": 0.4,
                    "cfg": 7.0,
                    "steps": 20,
                    "sampler": "DPM++ SDE Karras",
                    "mask_blur_px": 4,
                    "use_adetailer": True,
                    "ad_model": "face_yolov8n.pt",
                    "ad_confidence": 0.33,
                    "use_controlnet2": True,
                    "controlnet2_model": "xinsirControlnetCanny_v20",
                    "controlnet2_weight": 0.44
                }
            }
            
            # Save default config
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            return default_config
        
        # Load existing config
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
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
        # Placeholder enhancement - adjust saturation and sharpness
        enhanced = face_patch.copy()
        
        # Convert to float for processing
        if enhanced.dtype == np.uint8:
            enhanced = enhanced.astype(np.float32) / 255.0
        
        # Basic enhancement - increase saturation
        if len(enhanced.shape) == 3:
            # Convert to HSV for saturation adjustment
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 1)  # Increase saturation
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Convert back to uint8
        if face_patch.dtype == np.uint8:
            enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
    
    def integrate_with_detection(self, image: np.ndarray, 
                               detections: List[Dict[str, Any]]) -> np.ndarray:
        """Integrate B-Pass results back into the original image.
        
        Args:
            image: Original image
            detections: Face detections with enhanced patches
            
        Returns:
            Image with integrated B-Pass results
        """
        if not self.enabled or not self._initialized:
            return image
        
        # Process the entire image through B-Pass
        enhanced_patches = self.process_faces([], image)
        
        if enhanced_patches and len(enhanced_patches) > 0:
            # Use the first enhanced patch (should be the full enhanced image)
            return enhanced_patches[0]
        
        return image
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the B-Pass pipeline.
        
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
            'output_dir': self.output_dir,
            'config_path': self.config_path,
            'portrait_enhancer_available': PORTRAIT_ENHANCER_AVAILABLE
        }
    
    def is_available(self) -> bool:
        """Check if B-Pass pipeline is available.
        
        Returns:
            True if available, False otherwise
        """
        return self.enabled and self._initialized
