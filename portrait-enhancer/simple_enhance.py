#!/usr/bin/env python3
"""
Simple image enhancement using ADetailer 2CN Plus for face detection
and basic image processing without requiring WebUI.
Enhanced with smart fallback logic: 2+ faces = basic processing, 1 face = AD2CN Plus
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import time

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

def count_faces_basic(image_path):
    """Count faces using OpenCV for fallback decision"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load OpenCV face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return len(faces)
        
    except Exception as e:
        print(f"Error in face counting for {image_path}: {e}")
        return 0

def enhance_image_simple(image_path, output_path):
    """Simple image enhancement without AI models - used for 2+ faces"""
    try:
        print(f"  🎯 Using basic enhancement (2+ faces detected)")
        
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
    """Enhanced image processing with ADetailer 2CN Plus - used for 1 face"""
    try:
        print(f"  🎯 Using ADetailer 2CN Plus (1 face detected)")
        
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

def process_batch_smart(input_dir, output_dir, use_ad2cn=True):
    """Smart batch processing with intelligent fallback logic"""
    
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
    print("🎯 Smart processing logic: 2+ faces = basic enhancement, 1 face = AD2CN Plus")
    
    if use_ad2cn and AD2CN_AVAILABLE:
        print("✓ ADetailer 2CN Plus available for single face processing")
        
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
            print(f"⚠ Config loading failed: {e}, using basic processing for all")
            use_ad2cn = False
            config = None
    else:
        print("⚠ Using basic image processing for all images")
        config = None
    
    # Process each image
    successful = 0
    basic_processing_count = 0
    ad2cn_processing_count = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n=== [{i}/{len(image_files)}] {image_path} ===")
        
        # First, count faces to decide processing method
        face_count = count_faces_basic(image_path)
        print(f"  🔍 Detected {face_count} face(s)")
        
        # Create output path
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
        
        # Smart processing decision
        if face_count >= 2:
            # 2+ faces: use basic processing
            print(f"  📊 Multiple faces detected ({face_count}), using basic enhancement")
            success = enhance_image_simple(image_path, output_path)
            if success:
                basic_processing_count += 1
        elif face_count == 1 and use_ad2cn and AD2CN_AVAILABLE and config:
            # 1 face: use ADetailer 2CN Plus
            print(f"  🎯 Single face detected, using ADetailer 2CN Plus")
            try:
                success = enhance_image_with_ad2cn(image_path, output_path, config)
                if success:
                    ad2cn_processing_count += 1
            except Exception as e:
                print(f"  ADetailer 2CN Plus failed: {e}")
                print("  Falling back to basic processing...")
                success = enhance_image_simple(image_path, output_path)
                if success:
                    basic_processing_count += 1
        else:
            # No faces or fallback: use basic processing
            print(f"  📊 No faces or fallback mode, using basic enhancement")
            success = enhance_image_simple(image_path, output_path)
            if success:
                basic_processing_count += 1
        
        if success:
            successful += 1
            print(f"✅ Enhanced: {output_path}")
        else:
            print(f"❌ Failed to process: {image_path}")
    
    # Final summary
    print(f"\n🎉 Processing complete! {successful}/{len(image_files)} images processed successfully")
    print(f"📁 Enhanced images saved to: {output_dir}")
    print(f"\n📊 Processing Statistics:")
    print(f"  Basic enhancement (2+ faces): {basic_processing_count} images")
    print(f"  AD2CN Plus (1 face): {ad2cn_processing_count} images")
    print(f"  Total successful: {successful} images")

def process_batch(input_dir, output_dir, use_ad2cn=True):
    """Legacy batch processing - kept for compatibility"""
    print("⚠ Using legacy processing method. Consider using --smart flag for better results.")
    process_batch_smart(input_dir, output_dir, use_ad2cn)

def main():
    parser = argparse.ArgumentParser(description="Smart image enhancement with intelligent fallback logic")
    parser.add_argument("--input-dir", default="input", help="Input directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--use-ad2cn", action="store_true", help="Use ADetailer 2CN Plus")
    parser.add_argument("--smart", action="store_true", help="Use smart processing logic (recommended)")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process images with smart logic
    if args.smart:
        process_batch_smart(args.input_dir, args.output_dir, args.use_ad2cn)
    else:
        process_batch(args.input_dir, args.output_dir, args.use_ad2cn)

if __name__ == "__main__":
    main()
