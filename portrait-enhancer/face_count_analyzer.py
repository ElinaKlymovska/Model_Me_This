#!/usr/bin/env python3
"""
Enhanced Face Count Analyzer - Analyze face detection with detailed detector information
and recommended processing method based on face count
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import time

# Add adetailer_2cn_plus to path
ad2cn_path = Path(__file__).parent.parent / "adetailer_2cn_plus"
if ad2cn_path.exists():
    sys.path.insert(0, str(ad2cn_path))

try:
    from ad2cn.config import Config
    from ad2cn.pipeline.detect import DetectionPipeline
    from ad2cn.utils.io import load_image
    AD2CN_AVAILABLE = True
    print("✓ ADetailer 2CN Plus integration available")
except ImportError:
    AD2CN_AVAILABLE = False
    print("⚠ ADetailer 2CN Plus not available")

def count_faces_basic(image_path):
    """Basic face detection using OpenCV with timing"""
    start_time = time.time()
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return 0, "OpenCV", time.time() - start_time, "Failed to load image"
        
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
        
        detection_time = time.time() - start_time
        return len(faces), "OpenCV", detection_time, f"Found {len(faces)} face(s)"
        
    except Exception as e:
        detection_time = time.time() - start_time
        return 0, "OpenCV", detection_time, f"Error: {str(e)}"

def count_faces_ad2cn(image_path, config):
    """Advanced face detection using ADetailer 2CN Plus with detailed info"""
    start_time = time.time()
    try:
        # Load image
        img = load_image(image_path)
        if img is None:
            return 0, "AD2CN", time.time() - start_time, "Failed to load image"
        
        # Initialize detection pipeline
        pipeline = DetectionPipeline(config)
        
        # Detect faces
        detections = pipeline.detect_faces(img)
        
        detection_time = time.time() - start_time
        
        if detections:
            # Get detector information
            detector_info = []
            for detection in detections:
                if hasattr(detection, 'detector_name'):
                    detector_info.append(detection.detector_name)
                else:
                    detector_info.append("Unknown")
            
            # Count unique detectors used
            unique_detectors = list(set(detector_info))
            detector_summary = f"Used detectors: {', '.join(unique_detectors)}"
            
            return len(detections), "AD2CN", detection_time, f"Found {len(detections)} face(s) - {detector_summary}"
        else:
            return 0, "AD2CN", detection_time, "No faces detected"
        
    except Exception as e:
        detection_time = time.time() - start_time
        return 0, "AD2CN", detection_time, f"Error: {str(e)}"

def get_recommended_processing_method(face_count, ad2cn_available):
    """Get recommended processing method based on face count"""
    if face_count >= 2:
        return "Basic Enhancement", "📊 Multiple faces - use basic processing for stability"
    elif face_count == 1 and ad2cn_available:
        return "AD2CN Plus", "🎯 Single face - use ADetailer 2CN Plus for detailed enhancement"
    elif face_count == 1:
        return "Basic Enhancement", "🎯 Single face - AD2CN not available, using basic processing"
    else:
        return "Basic Enhancement", "📊 No faces - use basic processing"
    
def analyze_face_counts_enhanced(input_dir, output_dir, use_ad2cn=True):
    """Enhanced face count analysis with detector information and processing recommendations"""
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to analyze")
    
    # Initialize config if using AD2CN
    config = None
    if use_ad2cn and AD2CN_AVAILABLE:
        try:
            config_path = ad2cn_path / "config.yaml"
            if config_path.exists():
                config = Config()
                print("🎯 Using ADetailer 2CN Plus for face detection")
            else:
                config = Config()
                print("⚠ Using default AD2CN config")
        except Exception as e:
            print(f"⚠ AD2CN config failed: {e}, using basic detection")
            use_ad2cn = False
    else:
        print("⚠ Using basic OpenCV face detection")
    
    # Analyze each image
    total_faces = 0
    face_counts = []
    detector_stats = {}
    timing_stats = []
    processing_recommendations = {
        "Basic Enhancement": 0,
        "AD2CN Plus": 0
    }
    
    print("\n" + "="*100)
    print("ENHANCED FACE COUNT ANALYSIS WITH PROCESSING RECOMMENDATIONS")
    print("="*100)
    
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        
        # Count faces with detailed info
        if use_ad2cn and AD2CN_AVAILABLE and config:
            try:
                face_count, detector, detection_time, details = count_faces_ad2cn(image_path, config)
            except Exception as e:
                print(f"  AD2CN failed: {e}, falling back to basic")
                face_count, detector, detection_time, details = count_faces_basic(image_path)
        else:
            face_count, detector, detection_time, details = count_faces_basic(image_path)
        
        total_faces += face_count
        face_counts.append((filename, face_count, detector, detection_time, details))
        
        # Get processing recommendation
        recommended_method, reason = get_recommended_processing_method(face_count, use_ad2cn and AD2CN_AVAILABLE)
        processing_recommendations[recommended_method] += 1
        
        # Update detector statistics
        if detector not in detector_stats:
            detector_stats[detector] = {"count": 0, "faces": 0, "total_time": 0}
        detector_stats[detector]["count"] += 1
        detector_stats[detector]["faces"] += face_count
        detector_stats[detector]["total_time"] += detection_time
        
        timing_stats.append(detection_time)
        
        # Display result
        status = "✅" if face_count > 0 else "❌"
        print(f"{status} [{i:2d}/{len(image_files)}] {filename:<35} | "
              f"Faces: {face_count:2d} | {detector:<8} | {detection_time:.3f}s | "
              f"Method: {recommended_method:<18} | {reason}")
    
    # Summary
    print("="*100)
    print("DETAILED SUMMARY")
    print("="*100)
    print(f"Total images analyzed: {len(image_files)}")
    print(f"Total faces detected: {total_faces}")
    print(f"Average faces per image: {total_faces/len(image_files):.2f}")
    
    # Images with faces
    images_with_faces = sum(1 for _, count, _, _, _ in face_counts if count > 0)
    print(f"Images with faces: {images_with_faces}/{len(image_files)} ({images_with_faces/len(image_files)*100:.1f}%)")
    
    # Face count distribution
    face_distribution = {}
    for _, count, _, _, _ in face_counts:
        face_distribution[count] = face_distribution.get(count, 0) + 1
    
    print("\nFace count distribution:")
    for count in sorted(face_distribution.keys()):
        percentage = face_distribution[count] / len(image_files) * 100
        print(f"  {count} face(s): {face_distribution[count]} images ({percentage:.1f}%)")
    
    # Processing recommendations summary
    print("\nProcessing Method Recommendations:")
    print("-" * 40)
    for method, count in processing_recommendations.items():
        percentage = count / len(image_files) * 100
        print(f"  {method}: {count} images ({percentage:.1f}%)")
    
    # Detector performance analysis
    print("\nDetector Performance Analysis:")
    print("-" * 40)
    for detector, stats in detector_stats.items():
        avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        success_rate = stats["faces"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {detector}:")
        print(f"    Images processed: {stats['count']}")
        print(f"    Total faces found: {stats['faces']}")
        print(f"    Average time: {avg_time:.3f}s")
        print(f"    Success rate: {success_rate:.2f} faces per image")
    
    # Timing statistics
    if timing_stats:
        avg_time = sum(timing_stats) / len(timing_stats)
        min_time = min(timing_stats)
        max_time = max(timing_stats)
        print(f"\nTiming Statistics:")
        print(f"  Average detection time: {avg_time:.3f}s")
        print(f"  Fastest detection: {min_time:.3f}s")
        print(f"  Slowest detection: {max_time:.3f}s")
    
    # Save detailed results to file
    results_file = os.path.join(output_dir, "enhanced_face_count_analysis.txt")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("ENHANCED FACE COUNT ANALYSIS WITH PROCESSING RECOMMENDATIONS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total images analyzed: {len(image_files)}\n")
        f.write(f"Total faces detected: {total_faces}\n")
        f.write(f"Average faces per image: {total_faces/len(image_files):.2f}\n")
        f.write(f"Images with faces: {images_with_faces}/{len(image_files)} ({images_with_faces/len(image_files)*100:.1f}%)\n\n")
        
        f.write("PROCESSING RECOMMENDATIONS:\n")
        f.write("-" * 30 + "\n")
        for method, count in processing_recommendations.items():
            percentage = count / len(image_files) * 100
            f.write(f"{method}: {count} images ({percentage:.1f}%)\n")
        
        f.write(f"\nDETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        for filename, count, detector, time_taken, details in face_counts:
            recommended_method, reason = get_recommended_processing_method(count, use_ad2cn and AD2CN_AVAILABLE)
            f.write(f"{filename}: {count} face(s) | {detector} | {time_taken:.3f}s | {recommended_method} | {reason}\n")
        
        f.write(f"\nDetector Performance:\n")
        f.write("-" * 20 + "\n")
        for detector, stats in detector_stats.items():
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            success_rate = stats["faces"] / stats["count"] if stats["count"] > 0 else 0
            f.write(f"{detector}: {stats['count']} images, {stats['faces']} faces, avg time: {avg_time:.3f}s\n")
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    
    return total_faces, face_counts, detector_stats, processing_recommendations

def main():
    parser = argparse.ArgumentParser(description="Enhanced face count analysis with detector information and processing recommendations")
    parser.add_argument("--input-dir", default="input", help="Input directory with images")
    parser.add_argument("--output-dir", default="output", help="Output directory for results")
    parser.add_argument("--use-ad2cn", action="store_true", help="Use ADetailer 2CN Plus")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze face counts with enhanced information and recommendations
    analyze_face_counts_enhanced(args.input_dir, args.output_dir, args.use_ad2cn)

if __name__ == "__main__":
    main()
