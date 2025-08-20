#!/usr/bin/env python3
"""
CLI script for face detection and enhancement using ADetailer 2CN Plus.
Integrated with existing portrait-enhancer pipeline.
"""

import argparse
import sys
from pathlib import Path
import yaml
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ad2cn.config import Config
from ad2cn.pipeline.detect import DetectionPipeline
from ad2cn.utils.io import load_image, save_image, load_images_from_directory
from ad2cn.utils.vis import draw_detections, save_visualization
from ad2cn.utils.timing import Timer


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return Config(**config_data)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def process_single_image(image_path: str, pipeline: DetectionPipeline, 
                        output_path: str, enable_enhancement: bool = True,
                        show_visualization: bool = True):
    """Process a single image with optional enhancement."""
    print(f"Processing image: {image_path}")
    
    try:
        # Load image
        with Timer("image_loading"):
            image = load_image(image_path)
        
        print(f"Image loaded: {image.shape}")
        
        # Detect faces and optionally enhance
        with Timer("face_detection_and_enhancement"):
            detections = pipeline.detect_faces(image, enable_enhancement)
        
        print(f"Detected {len(detections)} faces")
        
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
        if output_path:
            if enable_enhancement:
                # Save enhanced results
                pipeline.save_enhanced_results(detections, output_path)
                
                # Save visualization if requested
                if show_visualization:
                    vis_path = Path(output_path) / "visualization.jpg"
                    save_visualization(image, str(vis_path), detections)
                    print(f"Visualization saved to: {vis_path}")
            else:
                # Save detection visualization only
                if show_visualization:
                    save_visualization(image, output_path, detections)
                else:
                    save_image(image, output_path)
                
                print(f"Results saved to: {output_path}")
        
        return detections
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []


def process_batch(input_dir: str, pipeline: DetectionPipeline, 
                  output_dir: str, enable_enhancement: bool = True,
                  show_visualization: bool = True):
    """Process a batch of images with optional enhancement."""
    print(f"Processing batch from: {input_dir}")
    
    try:
        # Load all images
        with Timer("batch_loading"):
            images = load_images_from_directory(input_dir)
        
        print(f"Loaded {len(images)} images")
        
        # Process each image
        all_results = []
        for i, image in enumerate(images):
            print(f"\nProcessing image {i+1}/{len(images)}")
            
            # Detect faces and optionally enhance
            with Timer("face_detection_and_enhancement"):
                detections = pipeline.detect_faces(image, enable_enhancement)
            
            print(f"  Detected {len(detections)} faces")
            
            # Save results
            if output_dir:
                image_output_dir = Path(output_dir) / f"image_{i:04d}"
                image_output_dir.mkdir(parents=True, exist_ok=True)
                
                if enable_enhancement:
                    # Save enhanced results
                    pipeline.save_enhanced_results(detections, str(image_output_dir))
                    
                    # Save visualization if requested
                    if show_visualization:
                        vis_path = image_output_dir / "visualization.jpg"
                        save_visualization(image, str(vis_path), detections)
                        print(f"  Visualization saved to: {vis_path}")
                else:
                    # Save detection visualization only
                    if show_visualization:
                        vis_path = image_output_dir / "detection.jpg"
                        save_visualization(image, str(vis_path), detections)
                        print(f"  Detection saved to: {vis_path}")
                
                print(f"  Results saved to: {image_output_dir}")
            
            all_results.append(detections)
        
        # Print summary
        total_faces = sum(len(detections) for detections in all_results)
        enhanced_count = sum(1 for detections in all_results 
                           for det in detections if det.get('b_pass_enhanced'))
        
        print(f"\nBatch processing complete!")
        print(f"Total faces detected: {total_faces}")
        if enable_enhancement:
            print(f"Images enhanced: {enhanced_count}")
        
        return all_results
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return []


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="ADetailer 2CN Plus - Face Detection and Enhancement CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image with enhancement
  python enhance.py -i image.jpg -o output/ -c config.yaml --enhance
  
  # Process single image without enhancement
  python enhance.py -i image.jpg -o output/ -c config.yaml --no-enhance
  
  # Process batch of images with enhancement
  python enhance.py -i input_dir/ -o output_dir/ -c config.yaml --enhance --batch
  
  # Show pipeline info
  python enhance.py -c config.yaml --info
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input image path or directory')
    parser.add_argument('-o', '--output',
                       help='Output path or directory')
    parser.add_argument('-c', '--config', required=True,
                       help='Configuration file path')
    parser.add_argument('--batch', action='store_true',
                       help='Process as batch (input should be directory)')
    parser.add_argument('--enhance', action='store_true',
                       help='Enable A-Pass and B-Pass enhancement (default)')
    parser.add_argument('--no-enhance', action='store_true',
                       help='Disable enhancement (detection only)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization overlays')
    parser.add_argument('--info', action='store_true',
                       help='Show pipeline information and exit')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize pipeline
    try:
        with Timer("pipeline_initialization"):
            pipeline = DetectionPipeline(config.pipeline.dict())
        
        print("Pipeline initialized successfully")
        
        # Show pipeline info if requested
        if args.info:
            pipeline_info = pipeline.get_pipeline_info()
            print("\nPipeline Information:")
            for key, value in pipeline_info.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
            return
        
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)
    
    # Determine enhancement setting
    enable_enhancement = not args.no_enhance  # Default to True unless --no-enhance
    
    if enable_enhancement:
        print("✓ Enhancement enabled (A-Pass + B-Pass)")
    else:
        print("⚠ Enhancement disabled (detection only)")
    
    # Process input
    if args.batch:
        # Batch processing
        if not Path(args.input).is_dir():
            print("Error: Input must be a directory for batch processing")
            sys.exit(1)
        
        if not args.output:
            print("Error: Output directory required for batch processing")
            sys.exit(1)
        
        # Create output directory
        Path(args.output).mkdir(parents=True, exist_ok=True)
        
        process_batch(args.input, pipeline, args.output, enable_enhancement, not args.no_viz)
        
    else:
        # Single image processing
        if not Path(args.input).is_file():
            print("Error: Input must be a file for single image processing")
            sys.exit(1)
        
        process_single_image(args.input, pipeline, args.output, enable_enhancement, not args.no_viz)


if __name__ == "__main__":
    main()
