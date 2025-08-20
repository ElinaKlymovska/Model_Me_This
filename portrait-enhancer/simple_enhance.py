#!/usr/bin/env python3
"""
Simple image enhancement using ADetailer 2CN Plus for face detection
and basic image processing without requiring WebUI.
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Add adetailer_2cn_plus to path
ad2cn_path = Path(__file__).parent.parent / "adetailer_2cn_plus"
if ad2cn_path.exists():
    sys.path.insert(0, str(ad2cn_path))

try:
    from ad2cn.config import Config
    from ad2cn.pipeline.detect import DetectionPipeline
    from ad2cn.utils.io import load_image, save_image
    AD2CN_AVAILABLE = True
    print("✓ ADetailer 2CN Plus integration available")
except ImportError:
    AD2CN_AVAILABLE = False
    print("⚠ ADetailer 2CN Plus not available, using basic processing")

def enhance_image_simple(image_path, output_path):
    """Simple image enhancement without AI models"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Basic enhancements
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)  # Increase sharpness
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)  # Increase contrast
        
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)  # Slight color enhancement
        
        # Save enhanced image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, quality=95)
        
        return True
    except Exception as e:
        print(f"Error enhancing {image_path}: {e}")
        return False

def enhance_image_with_ad2cn(image_path, output_path, config):
    """Enhanced image processing with ADetailer 2CN Plus"""
    try:
        # Load image
        img = load_image(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return False
        
        # Initialize detection pipeline
        pipeline = DetectionPipeline(config)
        
        # Detect faces
        print(f"  Detecting faces in {os.path.basename(image_path)}...")
        detections = pipeline.detect_faces(img)
        
        if detections:
            print(f"  Found {len(detections)} faces")
            
            # Apply face-specific enhancements
            enhanced_img = img.copy()
            
            for detection in detections:
                bbox = detection.bbox
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                
                # Extract face region
                face_region = enhanced_img[y1:y2, x1:x2]
                
                # Apply face enhancements
                # Increase sharpness in face region
                face_enhanced = cv2.GaussianBlur(face_region, (0, 0), 0.5)
                face_enhanced = cv2.addWeighted(face_region, 1.5, face_enhanced, -0.5, 0)
                
                # Replace face region
                enhanced_img[y1:y2, x1:x2] = face_enhanced
            
            # Save enhanced image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_image(enhanced_img, output_path)
            
            return True
        else:
            print(f"  No faces detected, using basic enhancement")
            return enhance_image_simple(image_path, output_path)
            
    except Exception as e:
        print(f"Error with ADetailer 2CN Plus: {e}")
        print("Falling back to basic processing...")
        return enhance_image_simple(image_path, output_path)

def process_batch(input_dir, output_dir, use_ad2cn=True):
    """Process all images in input directory"""
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    if use_ad2cn and AD2CN_AVAILABLE:
        print("🎯 Using ADetailer 2CN Plus for enhanced processing")
        
        # Try to load config, fallback to basic if failed
        try:
            config_path = ad2cn_path / "config.yaml"
            if config_path.exists():
                # Try different ways to load config
                try:
                    config = Config.from_yaml(str(config_path))
                except AttributeError:
                    # Fallback: create default config
                    config = Config()
                    print("⚠ Using default config (from_yaml not available)")
            else:
                config = Config()
                print("⚠ Using default config")
        except Exception as e:
            print(f"⚠ Config loading failed: {e}, using basic processing")
            use_ad2cn = False
            config = None
    else:
        print("⚠ Using basic image processing")
        config = None
    
    # Process each image
    successful = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"\n=== [{i}/{len(image_files)}] {image_path} ===")
        
        # Create output path
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
        
        # Process image
        if use_ad2cn and AD2CN_AVAILABLE and config:
            try:
                success = enhance_image_with_ad2cn(image_path, output_path, config)
            except Exception as e:
                print(f"  ADetailer 2CN Plus failed: {e}")
                print("  Falling back to basic processing...")
                success = enhance_image_simple(image_path, output_path)
        else:
            success = enhance_image_simple(image_path, output_path)
        
        if success:
            successful += 1
            print(f"✅ Enhanced: {output_path}")
        else:
            print(f"❌ Failed to process: {image_path}")
    
    print(f"\n🎉 Processing complete! {successful}/{len(image_files)} images processed successfully")
    print(f"📁 Enhanced images saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Simple image enhancement with ADetailer 2CN Plus")
    parser.add_argument("--input-dir", default="input", help="Input directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--use-ad2cn", action="store_true", help="Use ADetailer 2CN Plus")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process images
    process_batch(args.input_dir, args.output_dir, args.use_ad2cn)

if __name__ == "__main__":
    main()
