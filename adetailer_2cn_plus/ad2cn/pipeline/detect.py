"""
Main detection pipeline orchestrator with A-Pass and B-Pass integration.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2

from ..detectors.base import FaceDetector
from ..detectors.blazeface import BlazeFaceDetector
from ..detectors.retinaface import RetinaFaceDetector
from ..detectors.mtcnn import MTCNNDetector
from ..search.sliding_window import SlidingWindowSearch
from ..search.multi_scale import MultiScaleSearch
from ..alignment.align import FaceAligner
from ..utils.bbox import apply_nms
from ..utils.timing import Timer


class DetectionPipeline:
    """Main face detection pipeline orchestrator with enhancement integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize detection pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.detectors = {}
        self.search_strategy = None
        self.aligner = None
        self.a_pass_pipeline = None
        self.b_pass_pipeline = None
        self._initialized = False
        
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup pipeline components."""
        # Initialize detectors
        self._setup_detectors()
        
        # Setup search strategy
        self._setup_search_strategy()
        
        # Setup face aligner
        self.aligner = FaceAligner(
            desired_size=256,
            desired_face_ratio=0.75
        )
        
        # Setup enhancement pipelines
        self._setup_enhancement_pipelines()
        
        self._initialized = True
    
    def _setup_detectors(self):
        """Setup face detectors."""
        detector_configs = self.config.get('detectors', [])
        
        for det_config in detector_configs:
            detector_name = det_config['name']
            
            if detector_name == 'blazeface':
                self.detectors[detector_name] = BlazeFaceDetector(det_config)
            elif detector_name == 'retinaface':
                self.detectors[detector_name] = RetinaFaceDetector(det_config)
            elif detector_name == 'mtcnn':
                self.detectors[detector_name] = MTCNNDetector(det_config)
            else:
                raise ValueError(f"Unknown detector: {detector_name}")
    
    def _setup_search_strategy(self):
        """Setup search strategy."""
        search_config = self.config.get('search', {})
        strategy = search_config.get('strategy', 'sliding_window')
        
        if strategy == 'sliding_window':
            self.search_strategy = SlidingWindowSearch(
                window_size=search_config.get('window_size', 512),
                stride=search_config.get('stride', 256)
            )
        elif strategy == 'multi_scale':
            self.search_strategy = MultiScaleSearch(
                scale_factors=search_config.get('scale_factors', [1.0, 0.75, 0.5])
            )
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")
    
    def _setup_enhancement_pipelines(self):
        """Setup A-Pass and B-Pass enhancement pipelines."""
        # Setup A-Pass pipeline
        a_pass_config = self.config.get('a_pass', {})
        if a_pass_config.get('enabled', False):
            from .a_pass import APassPipeline
            self.a_pass_pipeline = APassPipeline(a_pass_config)
            print("✓ A-Pass pipeline initialized")
        
        # Setup B-Pass pipeline
        b_pass_config = self.config.get('b_pass', {})
        if b_pass_config.get('enabled', False):
            from .b_pass import BPassPipeline
            self.b_pass_pipeline = BPassPipeline(b_pass_config)
            print("✓ B-Pass pipeline initialized")
    
    def detect_faces(self, image: np.ndarray, 
                    enable_enhancement: bool = True) -> List[Dict[str, Any]]:
        """Detect faces using the pipeline with optional enhancement.
        
        Args:
            image: Input image
            enable_enhancement: Whether to run A-Pass and B-Pass enhancement
            
        Returns:
            List of face detections with optional enhancement results
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized")
        
        with Timer("face_detection") as timer:
            # Run detection based on configuration
            if self.config.get('enable_cascade', True):
                detections = self._cascade_detect(image)
            else:
                detections = self._single_detect(image)
            
            # Apply NMS if multiple detectors
            if len(self.detectors) > 1:
                detections = apply_nms(detections, self.config.get('nms_threshold', 0.3))
            
            # Extract face patches
            if detections:
                face_patches = self.aligner.extract_face_patches(image, detections)
                
                # Add face patches to detections
                for i, detection in enumerate(detections):
                    if i < len(face_patches):
                        detection['face_patch'] = face_patches[i]
                
                # Run enhancement if enabled
                if enable_enhancement:
                    detections = self._run_enhancement_pipeline(image, detections)
            
            # Add timing information
            for detection in detections:
                detection['processing_time'] = timer.elapsed_time
            
            return detections
    
    def _run_enhancement_pipeline(self, image: np.ndarray, 
                                detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run A-Pass and B-Pass enhancement pipelines.
        
        Args:
            image: Original image
            detections: Face detections
            
        Returns:
            Enhanced detections
        """
        enhanced_detections = detections.copy()
        
        # Run A-Pass enhancement
        if self.a_pass_pipeline and self.a_pass_pipeline.is_available():
            with Timer("a_pass_enhancement"):
                print("Running A-Pass enhancement...")
                a_pass_results = self.a_pass_pipeline.process_faces([], image)
                
                # Integrate A-Pass results
                if a_pass_results:
                    enhanced_image = self.a_pass_pipeline.integrate_with_detection(image, detections)
                    
                    # Update detections with A-Pass results
                    for detection in enhanced_detections:
                        detection['a_pass_enhanced'] = True
                        detection['a_pass_image'] = enhanced_image
                    
                    print(f"✓ A-Pass enhancement completed")
        
        # Run B-Pass enhancement
        if self.b_pass_pipeline and self.b_pass_pipeline.is_available():
            with Timer("b_pass_enhancement"):
                print("Running B-Pass enhancement...")
                b_pass_results = self.b_pass_pipeline.process_faces([], image)
                
                # Integrate B-Pass results
                if b_pass_results:
                    final_enhanced_image = self.b_pass_pipeline.integrate_with_detection(image, enhanced_detections)
                    
                    # Update detections with B-Pass results
                    for detection in enhanced_detections:
                        detection['b_pass_enhanced'] = True
                        detection['final_enhanced_image'] = final_enhanced_image
                    
                    print(f"✓ B-Pass enhancement completed")
        
        return enhanced_detections
    
    def _cascade_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run cascade detection using multiple detectors.
        
        Args:
            image: Input image
            
        Returns:
            List of detections
        """
        cascade_order = self.config.get('cascade_order', list(self.detectors.keys()))
        all_detections = []
        
        for detector_name in cascade_order:
            if detector_name not in self.detectors:
                continue
            
            detector = self.detectors[detector_name]
            
            # Load model if not loaded
            if not detector.is_initialized():
                detector.load_model()
            
            # Detect faces
            if self.search_strategy:
                detector_detections = self.search_strategy.search(image, detector)
            else:
                detector_detections = detector(image)
            
            # Add detector information
            for detection in detector_detections:
                detection['cascade_stage'] = detector_name
                detection['detector'] = detector_name
            
            all_detections.extend(detector_detections)
        
        return all_detections
    
    def _single_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run single detector detection.
        
        Args:
            image: Input image
            
        Returns:
            List of detections
        """
        # Use first available detector
        detector_name = list(self.detectors.keys())[0]
        detector = self.detectors[detector_name]
        
        # Load model if not loaded
        if not detector.is_initialized():
            detector.load_model()
        
        # Detect faces
        if self.search_strategy:
            detections = self.search_strategy.search(image, detector)
        else:
            detections = detector(image)
        
        # Add detector information
        for detection in detections:
            detection['detector'] = detector_name
        
        return detections
    
    def extract_faces(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Extract aligned face patches.
        
        Args:
            image: Input image
            detections: List of face detections
            
        Returns:
            List of aligned face patches
        """
        return self.aligner.extract_face_patches(image, detections)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline.
        
        Returns:
            Pipeline information
        """
        info = {
            'detectors': list(self.detectors.keys()),
            'search_strategy': self.search_strategy.__class__.__name__,
            'cascade_enabled': self.config.get('enable_cascade', True),
            'cascade_order': self.config.get('cascade_order', []),
            'initialized': self._initialized
        }
        
        # Add enhancement pipeline info
        if self.a_pass_pipeline:
            info['a_pass'] = self.a_pass_pipeline.get_pipeline_info()
        
        if self.b_pass_pipeline:
            info['b_pass'] = self.b_pass_pipeline.get_pipeline_info()
        
        return info
    
    def process_batch(self, images: List[np.ndarray], 
                     enable_enhancement: bool = True) -> List[List[Dict[str, Any]]]:
        """Process a batch of images.
        
        Args:
            images: List of input images
            enable_enhancement: Whether to run enhancement pipelines
            
        Returns:
            List of detection results for each image
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                detections = self.detect_faces(image, enable_enhancement)
                results.append(detections)
            except Exception as e:
                # Log error and continue with empty results
                print(f"Error processing image {i}: {e}")
                results.append([])
        
        return results
    
    def save_enhanced_results(self, detections: List[Dict[str, Any]], 
                            output_dir: str = "output"):
        """Save enhanced results from the pipeline.
        
        Args:
            detections: List of detections with enhancement results
            output_dir: Output directory for saving results
        """
        from pathlib import Path
        from ..utils.io import save_image
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, detection in enumerate(detections):
            # Save A-Pass result if available
            if detection.get('a_pass_enhanced') and 'a_pass_image' in detection:
                a_pass_path = output_path / f"a_pass_{i:04d}.png"
                save_image(detection['a_pass_image'], a_pass_path)
                print(f"Saved A-Pass result: {a_pass_path}")
            
            # Save B-Pass result if available
            if detection.get('b_pass_enhanced') and 'final_enhanced_image' in detection:
                b_pass_path = output_path / f"b_pass_{i:04d}.png"
                save_image(detection['final_enhanced_image'], b_pass_path)
                print(f"Saved B-Pass result: {b_pass_path}")
