#!/usr/bin/env python3
"""
Basic usage example for ADetailer 2CN Plus.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ad2cn.config import Config
from ad2cn.pipeline.detect import DetectionPipeline
from ad2cn.utils.io import load_image, save_image
from ad2cn.utils.vis import draw_detections, create_detection_summary
from ad2cn.utils.timing import Timer


def main():
    """Basic usage example."""
    print("ADetailer 2CN Plus - Basic Usage Example")
    print("=" * 50)
    
    # Create a minimal configuration
    config_data = {
        "pipeline": {
            "detectors": [
                {
                    "name": "blazeface",
                    "confidence_threshold": 0.5,
                    "nms_threshold": 0.3,
                    "max_faces": 10
                }
            ],
            "search": {
                "strategy": "multi_scale",
                "scale_factors": [1.0, 0.75, 0.5]
            },
            "enable_cascade": False
        }
    }
    
    try:
        # Initialize configuration
        config = Config(**config_data)
        print("✓ Configuration loaded successfully")
        
        # Initialize pipeline
        with Timer("pipeline_initialization"):
            pipeline = DetectionPipeline(config.pipeline.dict())
        print("✓ Pipeline initialized successfully")
        
        # Show pipeline information
        pipeline_info = pipeline.get_pipeline_info()
        print(f"\nPipeline Information:")
        for key, value in pipeline_info.items():
            print(f"  {key}: {value}")
        
        # Example: Process a sample image (if available)
        sample_image_path = Path(__file__).parent.parent / "tests" / "test_data" / "sample.jpg"
        
        if sample_image_path.exists():
            print(f"\n✓ Found sample image: {sample_image_path}")
            
            # Load image
            with Timer("image_loading"):
                image = load_image(sample_image_path)
            print(f"✓ Image loaded: {image.shape}")
            
            # Detect faces
            with Timer("face_detection"):
                detections = pipeline.detect_faces(image)
            print(f"✓ Detected {len(detections)} faces")
            
            # Show detection details
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                confidence = detection.get('confidence', 0.0)
                detector = detection.get('detector', 'unknown')
                print(f"  Face {i+1}: {detector} - confidence: {confidence:.3f}, bbox: {bbox}")
            
            # Save visualization
            output_path = Path(__file__).parent / "detection_result.jpg"
            save_image(draw_detections(image, detections), output_path)
            print(f"✓ Visualization saved to: {output_path}")
            
            # Create summary plot
            fig = create_detection_summary(detections)
            summary_path = Path(__file__).parent / "detection_summary.png"
            fig.savefig(summary_path, dpi=150, bbox_inches='tight')
            print(f"✓ Summary plot saved to: {summary_path}")
            
        else:
            print(f"\n⚠ Sample image not found: {sample_image_path}")
            print("  To test with an image, add a 'sample.jpg' to the test_data directory")
        
        print("\n✓ Example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis might be due to:")
        print("  - Missing dependencies")
        print("  - GPU not available")
        print("  - Model files not found")
        print("\nCheck the README.md for installation instructions.")


if __name__ == "__main__":
    main()
