#!/usr/bin/env python3
"""
Enhanced batch processing with ADetailer 2CN Plus integration.
This script provides better face detection and enhancement capabilities.
"""

import os
import argparse
import subprocess
import sys
import yaml
import glob
from pathlib import Path

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


def load_config(path):
    """Load configuration from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_ad2cn_config():
    """Create ADetailer 2CN Plus configuration."""
    return {
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
        }
    }


def process_with_ad2cn(image_path, config, work_dir, output_dir):
    """Process image using ADetailer 2CN Plus."""
    try:
        # Load image
        image = load_image(image_path)
        print(f"  Image loaded: {image.shape}")
        
        # Initialize pipeline
        ad2cn_config = create_ad2cn_config()
        pipeline = DetectionPipeline(ad2cn_config["pipeline"])
        
        # Detect faces with enhancement
        print("  Running enhanced face detection...")
        detections = pipeline.detect_faces(image, enable_enhancement=True)
        
        print(f"  Detected {len(detections)} faces with enhanced detection")
        
        # Save enhanced results
        pipeline.save_enhanced_results(detections, output_dir)
        
        return True
        
    except Exception as e:
        print(f"  Error with ADetailer 2CN Plus: {e}")
        return False


def main():
    """Main processing function."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--use-ad2cn", action="store_true", 
                   help="Use ADetailer 2CN Plus for enhanced processing")
    args = ap.parse_args()
    
    cfg = load_config(args.config)
    
    input_dir = cfg["io"]["input_dir"]
    work_dir = cfg["io"]["work_dir"]
    out_dir = cfg["io"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    # Find input images
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        imgs += glob.glob(os.path.join(input_dir, ext))
    imgs.sort()
    
    if not imgs:
        print(f"No images in {input_dir}")
        return
    
    print(f"Found {len(imgs)} images to process")
    
    if args.use_ad2cn and AD2CN_AVAILABLE:
        print("🎯 Using ADetailer 2CN Plus for enhanced processing")
    else:
        print("📝 Using standard processing pipeline")
    
    # Process each image
    for i, img in enumerate(imgs, 1):
        print(f"\n=== [{i}/{len(imgs)}] {img} ===")
        name = os.path.splitext(os.path.basename(img))[0]
        wd = os.path.join(work_dir, name)
        os.makedirs(os.path.join(wd, "a_pass"), exist_ok=True)
        os.makedirs(os.path.join(wd, "masks"), exist_ok=True)
        
        if args.use_ad2cn and AD2CN_AVAILABLE:
            # Use ADetailer 2CN Plus
            success = process_with_ad2cn(img, cfg, wd, out_dir)
            if not success:
                print("  Falling back to standard processing...")
                # Fallback to standard processing
                subprocess.run([sys.executable, "run_a_pass.py", "--config", args.config, "--input", img, "--workdir", wd], check=True)
                subprocess.run([sys.executable, "run_b_pass.py", "--config", args.config, "--workdir", wd, "--output", out_dir], check=True)
        else:
            # Use standard processing
            subprocess.run([sys.executable, "run_a_pass.py", "--config", args.config, "--input", img, "--workdir", wd], check=True)
            subprocess.run([sys.executable, "run_b_pass.py", "--config", args.config, "--workdir", wd, "--output", out_dir], check=True)
    
    print("\n🎉 All processing complete!")
    if args.use_ad2cn and AD2CN_AVAILABLE:
        print("✨ Enhanced with ADetailer 2CN Plus")


if __name__ == "__main__":
    main()
