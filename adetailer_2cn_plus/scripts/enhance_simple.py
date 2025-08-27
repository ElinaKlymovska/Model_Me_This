#!/usr/bin/env python3
"""
Simple script for complete face detection and enhancement.
"""

import sys
from pathlib import Path
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ad2cn.config import Config
from ad2cn.pipeline.detect import DetectionPipeline
from ad2cn.postprocess.face_inpainting import FaceInpainting
from ad2cn.utils.io import load_image, save_image
from ad2cn.utils.vis import draw_detections


def main():
    """Process all images in input directory with face detection and enhancement."""
    
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "data/samples/Massy"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/enhanced_massy"
    config_path = sys.argv[3] if len(sys.argv) > 3 else "adetailer_2cn_plus/config.yaml"
    
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Config: {config_path}")
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        config = Config(**config_data)
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Initialize pipeline and enhancer
    try:
        pipeline = DetectionPipeline(config.pipeline.model_dump())
        # Pass full config dict to enhancer for facial_enhancement settings
        enhancer = FaceInpainting(config.model_dump())
        print("Pipeline and enhancer initialized")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    input_path = Path(input_dir)
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in input_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in supported_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    total_faces = 0
    
    for i, image_file in enumerate(sorted(image_files)):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {image_file.name}")
        
        try:
            # Load image
            image = load_image(str(image_file))
            print(f"  Loaded image: {image.shape}")
            
            # Detect faces
            detections = pipeline.detect(image)
            print(f"  Detected {len(detections)} faces")
            total_faces += len(detections)
            
            # Show detection details
            for j, detection in enumerate(detections):
                bbox = detection['bbox']
                confidence = detection.get('confidence', 0.0)
                detector = detection.get('detector', 'unknown')
                print(f"    Face {j+1}: {detector} - confidence: {confidence:.3f}")
            
            # Apply face enhancement if faces detected
            if detections:
                print(f"  Applying enhancement to {len(detections)} faces...")
                # Pass image path for person-specific contouring settings
                enhanced_image = enhancer.process_multiple_faces(image, detections, str(image_file))
                print(f"  ✓ Enhancement with contouring applied")
            else:
                enhanced_image = image.copy()
                print(f"  No faces to enhance")
            
            # Save enhanced image
            output_file = Path(output_dir) / f"enhanced_{image_file.stem}.jpg"
            save_image(enhanced_image, str(output_file))
            print(f"  ✓ Saved enhanced image: {output_file}")
            
            # Save detection visualization if faces found
            if detections:
                viz_file = Path(output_dir) / f"detected_{image_file.stem}.jpg"
                viz_image = draw_detections(image, detections)
                save_image(viz_image, str(viz_file))
                print(f"  ✓ Saved detection visualization: {viz_file}")
                
        except Exception as e:
            print(f"  ✗ Error processing {image_file.name}: {e}")
    
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE!")
    print(f"Images processed: {len(image_files)}")
    print(f"Total faces detected: {total_faces}")
    print(f"Total faces enhanced: {total_faces}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()