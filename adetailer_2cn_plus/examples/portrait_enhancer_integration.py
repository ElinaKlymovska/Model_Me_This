#!/usr/bin/env python3
"""
Example of integrating ADetailer 2CN Plus with existing portrait-enhancer pipeline.
This demonstrates how to use your existing run_a_pass.py and run_b_pass.py scripts.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ad2cn.config import Config
from ad2cn.pipeline.detect import DetectionPipeline
from ad2cn.utils.io import load_image, save_image
from ad2cn.utils.vis import draw_detections, create_detection_summary
from ad2cn.utils.timing import Timer


def create_integration_config():
    """Create configuration for portrait-enhancer integration."""
    config_data = {
        "pipeline": {
            "detectors": [
                {
                    "name": "blazeface",
                    "confidence_threshold": 0.5,
                    "nms_threshold": 0.3,
                    "max_faces": 100,
                    "input_size": 128
                },
                {
                    "name": "retinaface",
                    "confidence_threshold": 0.7,
                    "nms_threshold": 0.3,
                    "max_faces": 50,
                    "input_size": 640
                }
            ],
            "search": {
                "strategy": "sliding_window",
                "window_size": 512,
                "stride": 256
            },
            "enable_cascade": True,
            "cascade_order": ["blazeface", "retinaface"]
        },
        "a_pass": {
            "enabled": True,
            "workdir": "work",
            "batch_size": 4,
            "device": "auto"
        },
        "b_pass": {
            "enabled": True,
            "workdir": "work",
            "output_dir": "output",
            "config_path": "config.yaml",
            "batch_size": 4,
            "device": "auto"
        },
        "portrait_enhancer": {
            "enabled": True,
            "auto_detect": True,
            "path": "../portrait-enhancer",
            "use_existing_config": True,
            "fallback_to_basic": True
        }
    }
    
    return Config(**config_data)


def run_full_pipeline_example():
    """Run the full pipeline with portrait-enhancer integration."""
    print("🎭 ADetailer 2CN Plus + Portrait-Enhancer Integration Example")
    print("=" * 60)
    
    try:
        # Create configuration
        print("1. Creating configuration...")
        config = create_integration_config()
        print("✓ Configuration created")
        
        # Initialize pipeline
        print("\n2. Initializing pipeline...")
        with Timer("pipeline_initialization"):
            pipeline = DetectionPipeline(config.pipeline.dict())
        print("✓ Pipeline initialized")
        
        # Show pipeline information
        print("\n3. Pipeline Information:")
        pipeline_info = pipeline.get_pipeline_info()
        for key, value in pipeline_info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Check if we have a sample image
        sample_image_path = Path(__file__).parent.parent / "tests" / "test_data" / "sample.jpg"
        
        if not sample_image_path.exists():
            print(f"\n⚠ Sample image not found: {sample_image_path}")
            print("  To test with an image, add a 'sample.jpg' to the test_data directory")
            print("  Or provide your own image path below")
            
            # Ask for image path
            custom_path = input("\nEnter path to your image (or press Enter to skip): ").strip()
            if custom_path and Path(custom_path).exists():
                sample_image_path = Path(custom_path)
            else:
                print("No valid image provided. Exiting.")
                return
        
        # Process image
        print(f"\n4. Processing image: {sample_image_path}")
        
        # Load image
        with Timer("image_loading"):
            image = load_image(str(sample_image_path))
        print(f"✓ Image loaded: {image.shape}")
        
        # Run detection with enhancement
        print("\n5. Running face detection + enhancement...")
        with Timer("full_pipeline"):
            detections = pipeline.detect_faces(image, enable_enhancement=True)
        
        print(f"✓ Processing complete!")
        print(f"  Detected faces: {len(detections)}")
        
        # Show detection details
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection.get('confidence', 0.0)
            detector = detection.get('detector', 'unknown')
            enhancement_status = []
            
            if detection.get('a_pass_enhanced'):
                enhancement_status.append('A-Pass')
            if detection.get('b_pass_enhanced'):
                enhancement_status.append('B-Pass')
            
            status = f"enhanced: {', '.join(enhancement_status)}" if enhancement_status else "no enhancement"
            print(f"  Face {i+1}: {detector} - confidence: {confidence:.3f}, bbox: {bbox}, {status}")
        
        # Save results
        output_dir = Path("output") / "integration_example"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n6. Saving results to: {output_dir}")
        
        # Save enhanced results
        pipeline.save_enhanced_results(detections, str(output_dir))
        
        # Save visualization
        vis_path = output_dir / "visualization.jpg"
        save_visualization(image, str(vis_path), detections)
        print(f"✓ Visualization saved: {vis_path}")
        
        # Create and save detection summary
        summary_path = output_dir / "detection_summary.png"
        fig = create_detection_summary(detections, str(sample_image_path))
        fig.savefig(summary_path, dpi=150, bbox_inches='tight')
        print(f"✓ Detection summary saved: {summary_path}")
        
        # Show timing information
        print(f"\n7. Performance Summary:")
        print(f"  Pipeline initialization: {pipeline_info.get('initialization_time', 'N/A')}")
        print(f"  Image loading: {getattr(Timer.get_last('image_loading'), 'elapsed_time', 'N/A'):.3f}s")
        print(f"  Full pipeline: {getattr(Timer.get_last('full_pipeline'), 'elapsed_time', 'N/A'):.3f}s")
        
        print(f"\n🎉 Integration example completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
        
        # Check if enhancement was successful
        enhanced_count = sum(1 for det in detections if det.get('b_pass_enhanced'))
        if enhanced_count > 0:
            print(f"✨ {enhanced_count} images were enhanced using portrait-enhancer!")
        else:
            print("⚠ No images were enhanced. Check if portrait-enhancer is available.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure portrait-enhancer is installed at ../portrait-enhancer")
        print("2. Check that all dependencies are installed: make setup")
        print("3. Verify your image path is correct")
        print("4. Check the logs for more detailed error information")


def run_detection_only_example():
    """Run detection only without enhancement."""
    print("🔍 ADetailer 2CN Plus - Detection Only Example")
    print("=" * 50)
    
    try:
        # Create minimal configuration
        config_data = {
            "pipeline": {
                "detectors": [
                    {
                        "name": "blazeface",
                        "confidence_threshold": 0.5,
                        "nms_threshold": 0.3,
                        "max_faces": 100
                    }
                ],
                "search": {
                    "strategy": "sliding_window",
                    "window_size": 512,
                    "stride": 256
                },
                "enable_cascade": False
            },
            "a_pass": {"enabled": False},
            "b_pass": {"enabled": False}
        }
        
        config = Config(**config_data)
        
        # Initialize pipeline
        pipeline = DetectionPipeline(config.pipeline.dict())
        print("✓ Pipeline initialized (detection only)")
        
        # Check for sample image
        sample_image_path = Path(__file__).parent.parent / "tests" / "test_data" / "sample.jpg"
        
        if not sample_image_path.exists():
            print(f"⚠ Sample image not found: {sample_image_path}")
            print("Please add a sample.jpg to the test_data directory")
            return
        
        # Load and process image
        image = load_image(str(sample_image_path))
        detections = pipeline.detect_faces(image, enable_enhancement=False)
        
        print(f"✓ Detected {len(detections)} faces")
        
        # Save detection results
        output_dir = Path("output") / "detection_only"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        vis_path = output_dir / "detection.jpg"
        save_visualization(image, str(vis_path), detections)
        print(f"✓ Detection visualization saved: {vis_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function with menu selection."""
    print("🎭 ADetailer 2CN Plus Examples")
    print("=" * 40)
    print("1. Full pipeline with portrait-enhancer integration")
    print("2. Detection only (no enhancement)")
    print("3. Exit")
    
    while True:
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == "1":
            run_full_pipeline_example()
            break
        elif choice == "2":
            run_detection_only_example()
            break
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
