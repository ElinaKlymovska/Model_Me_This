#!/usr/bin/env python3
"""
Face Verification Script - Compare face counts between original and enhanced images
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

def count_faces_opencv(image_path):
    """Count faces using OpenCV Haar Cascade"""
    start_time = time.time()
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return 0, time.time() - start_time, "Failed to load image"
        
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
        return len(faces), detection_time, f"Found {len(faces)} face(s)"
        
    except Exception as e:
        detection_time = time.time() - start_time
        return 0, detection_time, f"Error: {str(e)}"

def verify_face_counts(original_dir, enhanced_dir, output_dir):
    """Verify face counts between original and enhanced images"""
    
    # Get all original image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    original_files = []
    
    for ext in image_extensions:
        original_files.extend(glob.glob(os.path.join(original_dir, ext)))
        original_files.extend(glob.glob(os.path.join(original_dir, ext.upper())))
    
    if not original_files:
        print(f"No original images found in {original_dir}")
        return
    
    print(f"Found {len(original_files)} original images to verify")
    
    # Analyze each image
    verification_results = []
    total_original_faces = 0
    total_enhanced_faces = 0
    
    print("\n" + "="*100)
    print("FACE COUNT VERIFICATION - ORIGINAL vs ENHANCED")
    print("="*100)
    
    for i, original_path in enumerate(original_files, 1):
        filename = os.path.basename(original_path)
        name, ext = os.path.splitext(filename)
        
        # Find corresponding enhanced image
        enhanced_filename = f"{name}_enhanced{ext}"
        enhanced_path = os.path.join(enhanced_dir, enhanced_filename)
        
        # Count faces in original
        original_faces, original_time, original_details = count_faces_opencv(original_path)
        total_original_faces += original_faces
        
        # Count faces in enhanced
        if os.path.exists(enhanced_path):
            enhanced_faces, enhanced_time, enhanced_details = count_faces_opencv(enhanced_path)
            total_enhanced_faces += enhanced_faces
            enhanced_status = "✅"
        else:
            enhanced_faces, enhanced_time, enhanced_details = 0, 0, "File not found"
            enhanced_status = "❌"
        
        # Check if face count changed
        face_count_changed = original_faces != enhanced_faces
        change_indicator = "🔄" if face_count_changed else "✅"
        
        verification_results.append({
            'filename': filename,
            'original_faces': original_faces,
            'enhanced_faces': enhanced_faces,
            'original_time': original_time,
            'enhanced_time': enhanced_time,
            'face_count_changed': face_count_changed,
            'enhanced_exists': os.path.exists(enhanced_path)
        })
        
        # Display result
        print(f"{change_indicator} [{i:2d}/{len(original_files)}] {filename:<35} | "
              f"Original: {original_faces:2d} faces ({original_time:.3f}s) | "
              f"{enhanced_status} Enhanced: {enhanced_faces:2d} faces ({enhanced_time:.3f}s) | "
              f"{'CHANGED' if face_count_changed else 'SAME'}")
    
    # Summary
    print("="*100)
    print("VERIFICATION SUMMARY")
    print("="*100)
    print(f"Total images analyzed: {len(original_files)}")
    print(f"Total faces in original images: {total_original_faces}")
    print(f"Total faces in enhanced images: {total_enhanced_faces}")
    
    # Face count changes
    images_with_changes = sum(1 for result in verification_results if result['face_count_changed'])
    print(f"Images with face count changes: {images_with_changes}/{len(original_files)} ({images_with_changes/len(original_files)*100:.1f}%)")
    
    # Enhanced images status
    enhanced_images_found = sum(1 for result in verification_results if result['enhanced_exists'])
    print(f"Enhanced images found: {enhanced_images_found}/{len(original_files)} ({enhanced_images_found/len(original_files)*100:.1f}%)")
    
    # Face count distribution analysis
    print("\nFace Count Distribution Analysis:")
    print("-" * 50)
    
    # Original images
    original_distribution = {}
    for result in verification_results:
        count = result['original_faces']
        original_distribution[count] = original_distribution.get(count, 0) + 1
    
    print("Original images:")
    for count in sorted(original_distribution.keys()):
        percentage = original_distribution[count] / len(original_files) * 100
        print(f"  {count} face(s): {original_distribution[count]} images ({percentage:.1f}%)")
    
    # Enhanced images
    enhanced_distribution = {}
    for result in verification_results:
        if result['enhanced_exists']:
            count = result['enhanced_faces']
            enhanced_distribution[count] = enhanced_distribution.get(count, 0) + 1
    
    print("\nEnhanced images:")
    for count in sorted(enhanced_distribution.keys()):
        percentage = enhanced_distribution[count] / enhanced_images_found * 100
        print(f"  {count} face(s): {enhanced_distribution[count]} images ({percentage:.1f}%)")
    
    # Performance comparison
    print("\nPerformance Comparison:")
    print("-" * 30)
    
    original_times = [result['original_time'] for result in verification_results]
    enhanced_times = [result['enhanced_time'] for result in verification_results if result['enhanced_exists']]
    
    if original_times:
        avg_original_time = sum(original_times) / len(original_times)
        print(f"Original images - Average detection time: {avg_original_time:.3f}s")
    
    if enhanced_times:
        avg_enhanced_time = sum(enhanced_times) / len(enhanced_times)
        print(f"Enhanced images - Average detection time: {avg_enhanced_time:.3f}s")
    
    # Save detailed results
    results_file = os.path.join(output_dir, "face_verification_results.txt")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("FACE COUNT VERIFICATION RESULTS\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total images analyzed: {len(original_files)}\n")
        f.write(f"Total faces in original: {total_original_faces}\n")
        f.write(f"Total faces in enhanced: {total_enhanced_faces}\n")
        f.write(f"Images with face count changes: {images_with_changes}\n")
        f.write(f"Enhanced images found: {enhanced_images_found}\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 20 + "\n")
        for result in verification_results:
            f.write(f"{result['filename']}: Original={result['original_faces']} faces, "
                   f"Enhanced={result['enhanced_faces']} faces, "
                   f"Changed={'Yes' if result['face_count_changed'] else 'No'}\n")
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    
    return verification_results

def main():
    parser = argparse.ArgumentParser(description="Verify face counts between original and enhanced images")
    parser.add_argument("--original-dir", default="input", help="Directory with original images")
    parser.add_argument("--enhanced-dir", default="output/all_images", help="Directory with enhanced images")
    parser.add_argument("--output-dir", default="output", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    if not os.path.exists(args.original_dir):
        print(f"❌ Original directory not found: {args.original_dir}")
        return
    
    if not os.path.exists(args.enhanced_dir):
        print(f"❌ Enhanced directory not found: {args.enhanced_dir}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify face counts
    verify_face_counts(args.original_dir, args.enhanced_dir, args.output_dir)

if __name__ == "__main__":
    main()
