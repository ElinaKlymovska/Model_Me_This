#!/usr/bin/env python3
"""
CLI script for face detection using ADetailer 2CN Plus.
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
                        output_path: str, show_visualization: bool = True):
    """Process a single image."""
    print(f"Processing image: {image_path}")
    
    try:
        # Load image
        with Timer("image_loading"):
            image = load_image(image_path)
        
        print(f"Image loaded: {image.shape}")
        
        # Detect faces
        with Timer("face_detection"):
            detections = pipeline.detect(image)
        
        print(f"Detected {len(detections)} faces")
        
        # Show detection details
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection.get('confidence', 0.0)
            detector = detection.get('detector', 'unknown')
            print(f"  Face {i+1}: {detector} - confidence: {confidence:.3f}, bbox: {bbox}")
        
        # Save results
        if output_path:
            # Save image with detections
            if show_visualization:
                save_visualization(image, output_path, detections)
            else:
                # Save original image
                save_image(image, output_path)
            
            print(f"Results saved to: {output_path}")
        
        return detections
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []


def process_batch(input_dir: str, pipeline: DetectionPipeline, 
                  output_dir: str, show_visualization: bool = True):
    """Process a batch of images."""
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
            
            # Detect faces
            with Timer("face_detection"):
                detections = pipeline.detect(image)
            
            print(f"  Detected {len(detections)} faces")
            
            # Save results
            if output_dir:
                output_path = Path(output_dir) / f"result_{i:04d}.jpg"
                
                if show_visualization:
                    save_visualization(image, str(output_path), detections)
                else:
                    save_image(image, str(output_path))
                
                print(f"  Saved to: {output_path}")
            
            all_results.append(detections)
        
        # Print summary
        total_faces = sum(len(detections) for detections in all_results)
        print(f"\nBatch processing complete. Total faces detected: {total_faces}")
        
        return all_results
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return []


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="ADetailer 2CN Plus - Face Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python detect.py -i image.jpg -o result.jpg -c config.yaml
  
  # Process batch of images
  python detect.py -i input_dir/ -o output_dir/ -c config.yaml --batch
  
  # Show pipeline info
  python detect.py -c config.yaml --info
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
            print("\nPipeline Information:")
            print(f"  Detectors: {list(pipeline.detectors.keys())}")
            print(f"  Config: {config.pipeline.dict()}")
            return
        
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)
    
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
        
        process_batch(args.input, pipeline, args.output, not args.no_viz)
        
    else:
        # Single image processing
        if not Path(args.input).is_file():
            print("Error: Input must be a file for single image processing")
            sys.exit(1)
        
        process_single_image(args.input, pipeline, args.output, not args.no_viz)


if __name__ == "__main__":
    main()
