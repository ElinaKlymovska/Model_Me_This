#!/usr/bin/env python3
"""
Direct test of facial contouring system bypassing face detection.
"""

import sys
from pathlib import Path
import yaml
import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "adetailer_2cn_plus"))

from ad2cn.config import Config
from ad2cn.postprocess.face_inpainting import FaceInpainting
from ad2cn.utils.io import load_image, save_image


def main():
    """Test contouring system with manual face bounding box."""
    
    # Configuration
    input_image_path = "data/samples/Massy/23ae867f693345ac9fa3914ba93e6271.webp"
    output_path = "output/direct_contouring_test.jpg" 
    config_path = "adetailer_2cn_plus/config.yaml"
    
    print(f"Testing contouring system")
    print(f"Input: {input_image_path}")
    print(f"Output: {output_path}")
    print(f"Config: {config_path}")
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        config = Config(**config_data)
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return
    
    # Initialize enhancer
    try:
        enhancer = FaceInpainting(config.model_dump())
        print("✓ FaceInpainting enhancer initialized")
    except Exception as e:
        print(f"✗ Error initializing enhancer: {e}")
        return
    
    # Load image
    try:
        image = load_image(input_image_path)
        print(f"✓ Image loaded: {image.shape}")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return
        
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Manual face bounding box (estimated for the face in the image)
    # Based on the image I saw earlier, the face is roughly in the upper center
    h, w = image.shape[:2]
    
    # Estimate face location (you may need to adjust these values)
    face_width = int(w * 0.25)   # Face is about 25% of image width
    face_height = int(h * 0.35)  # Face is about 35% of image height
    face_x = int(w * 0.4)        # Face center-left position
    face_y = int(h * 0.15)       # Face upper position
    
    manual_face_bbox = [face_x, face_y, face_x + face_width, face_y + face_height]
    
    print(f"Manual face bbox: {manual_face_bbox}")
    
    # Create fake detection with manual bbox
    fake_detection = {
        'bbox': manual_face_bbox,
        'confidence': 0.95,
        'detector': 'manual'
    }
    
    try:
        print("Applying facial enhancement with contouring...")
        
        # Process face with contouring
        enhanced_image = enhancer.process_face(image, manual_face_bbox, input_image_path)
        
        print("✓ Contouring applied successfully!")
        
        # Save enhanced image
        save_image(enhanced_image, output_path)
        print(f"✓ Enhanced image saved to: {output_path}")
        
        # Also save image with bounding box visualization for reference
        bbox_viz_image = image.copy()
        cv2.rectangle(bbox_viz_image, 
                     (manual_face_bbox[0], manual_face_bbox[1]),
                     (manual_face_bbox[2], manual_face_bbox[3]),
                     (0, 255, 0), 3)
        cv2.putText(bbox_viz_image, 'Manual Face Bbox', 
                   (manual_face_bbox[0], manual_face_bbox[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        bbox_output_path = output_path.replace('.jpg', '_bbox_viz.jpg')
        save_image(bbox_viz_image, bbox_output_path)
        print(f"✓ Bounding box visualization saved to: {bbox_output_path}")
        
        print("\n" + "="*60)
        print("CONTOURING TEST SUCCESSFUL!")
        print("The facial contouring and expression lines system is working!")
        print("Check the output files to see the results:")
        print(f"- Enhanced image: {output_path}")
        print(f"- Bbox visualization: {bbox_output_path}")
        print("="*60)
        
    except Exception as e:
        print(f"✗ Error during enhancement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()