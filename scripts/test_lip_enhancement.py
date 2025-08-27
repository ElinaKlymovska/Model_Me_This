#!/usr/bin/env python3
"""
Test enhanced lip contouring and perioral wrinkles system.
Tests the new lip enhancement features on person-specific samples.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import yaml
import argparse
from typing import Dict, List, Any

# Add the project root to path
sys.path.append(str(Path(__file__).parent.parent))

from adetailer_2cn_plus.ad2cn.postprocess.facial_contouring import ContouringMask
from adetailer_2cn_plus.ad2cn.postprocess.expression_lines import ExpressionProcessor
from adetailer_2cn_plus.ad2cn.pipeline.detect import DetectionPipeline


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_side_by_side_comparison(original: np.ndarray, enhanced: np.ndarray, 
                                   title: str = "Before vs After") -> np.ndarray:
    """Create side-by-side comparison image."""
    # Ensure both images have same height
    h1, w1 = original.shape[:2]
    h2, w2 = enhanced.shape[:2]
    
    if h1 != h2:
        target_h = min(h1, h2)
        original = cv2.resize(original, (int(w1 * target_h / h1), target_h))
        enhanced = cv2.resize(enhanced, (int(w2 * target_h / h2), target_h))
    
    # Create side-by-side image
    comparison = np.hstack([original, enhanced])
    
    # Add title
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    thickness = 2
    
    # Get text size
    text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
    text_x = (comparison.shape[1] - text_size[0]) // 2
    text_y = 30
    
    # Add black background for text
    cv2.rectangle(comparison, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    
    # Add text
    cv2.putText(comparison, title, (text_x, text_y), font, font_scale, font_color, thickness)
    
    # Add "BEFORE" and "AFTER" labels
    h, w = comparison.shape[:2]
    cv2.putText(comparison, "BEFORE", (w//4 - 40, h - 20), font, 0.6, (255, 255, 255), 2)
    cv2.putText(comparison, "AFTER", (3*w//4 - 35, h - 20), font, 0.6, (255, 255, 255), 2)
    
    return comparison


def test_lip_enhancement_for_person(person_name: str, image_path: Path, 
                                    config: Dict[str, Any], output_dir: Path) -> bool:
    """Test lip enhancement for a specific person."""
    
    print(f"\n=== Testing Lip Enhancement for {person_name.upper()} ===")
    
    # Load image
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return False
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    original_image = image.copy()
    print(f"Loaded image: {image_path.name} ({image.shape[1]}x{image.shape[0]})")
    
    # Initialize detection pipeline
    detect_pipeline = DetectionPipeline(config)
    
    # Detect faces
    detections = detect_pipeline.detect(image)
    if not detections:
        print("No faces detected!")
        return False
    
    print(f"Detected {len(detections)} face(s)")
    
    # Get person-specific settings
    person_config = config.get('facial_enhancement', {}).get('person_profiles', {}).get(person_name, {})
    if not person_config:
        print(f"No configuration found for person: {person_name}")
        return False
    
    lip_config = person_config.get('lip_enhancement', {})
    print(f"Lip enhancement settings: {lip_config}")
    
    # Initialize enhancement modules
    facial_contouring = ContouringMask(config)
    expression_lines = ExpressionProcessor(config)
    
    # Process each detected face
    enhanced_image = image.copy()
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        print(f"Processing face {i+1}: bbox=({x1},{y1},{x2},{y2}), size=({x2-x1}x{y2-y1})")
        
        # Apply facial contouring with person-specific settings
        enhanced_image = facial_contouring.apply_contouring(
            enhanced_image, bbox, 
            image_path=str(image_path), 
            landmarks=detection.get('landmarks')
        )
        
        # Apply expression lines
        from adetailer_2cn_plus.ad2cn.postprocess.expression_lines import ExpressionType
        enhanced_image = expression_lines.apply_expression_lines(
            enhanced_image, bbox,
            expression=ExpressionType.NEUTRAL,
            image_path=str(image_path)
        )
    
    # Create comparison image
    comparison = create_side_by_side_comparison(
        original_image, enhanced_image, 
        f"{person_name.upper()} - Lip Enhancement Test"
    )
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual images
    original_path = output_dir / f"{person_name}_original_{image_path.stem}.jpg"
    enhanced_path = output_dir / f"{person_name}_enhanced_{image_path.stem}.jpg"
    comparison_path = output_dir / f"{person_name}_comparison_{image_path.stem}.jpg"
    
    cv2.imwrite(str(original_path), original_image)
    cv2.imwrite(str(enhanced_path), enhanced_image)
    cv2.imwrite(str(comparison_path), comparison)
    
    print(f"Results saved:")
    print(f"  Original: {original_path}")
    print(f"  Enhanced: {enhanced_path}")
    print(f"  Comparison: {comparison_path}")
    
    return True


def test_all_persons(config: Dict[str, Any], input_dir: Path, output_dir: Path) -> None:
    """Test lip enhancement for all persons with available images."""
    
    persons = ['massy', 'orbi', 'yana']
    test_results = {}
    
    for person in persons:
        print(f"\n{'='*60}")
        print(f"TESTING {person.upper()}")
        print(f"{'='*60}")
        
        # Look for person's images
        person_dir = input_dir / person.capitalize()
        if not person_dir.exists():
            print(f"Directory not found: {person_dir}")
            test_results[person] = False
            continue
        
        # Get first few images for testing
        image_files = list(person_dir.glob("*.webp")) + list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        if not image_files:
            print(f"No images found in {person_dir}")
            test_results[person] = False
            continue
        
        # Test with first image
        success = test_lip_enhancement_for_person(person, image_files[0], config, output_dir)
        test_results[person] = success
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for person, success in test_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{person.capitalize()}: {status}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test enhanced lip contouring system')
    parser.add_argument('--config', '-c', 
                        default='adetailer_2cn_plus/config.yaml',
                        help='Configuration file path')
    parser.add_argument('--input', '-i',
                        default='data/samples',
                        help='Input directory with person subdirectories')
    parser.add_argument('--output', '-o',
                        default='output/lip_enhancement_test',
                        help='Output directory for results')
    parser.add_argument('--person', '-p',
                        help='Test specific person only (massy/orbi/yana)')
    
    args = parser.parse_args()
    
    # Convert paths
    config_path = Path(args.config)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    if args.person:
        # Test specific person
        person = args.person.lower()
        if person not in ['massy', 'orbi', 'yana']:
            print(f"Invalid person: {person}. Must be one of: massy, orbi, yana")
            sys.exit(1)
        
        person_dir = input_dir / person.capitalize()
        image_files = list(person_dir.glob("*.webp")) + list(person_dir.glob("*.jpg"))
        if image_files:
            test_lip_enhancement_for_person(person, image_files[0], config, output_dir)
        else:
            print(f"No images found for {person}")
    else:
        # Test all persons
        test_all_persons(config, input_dir, output_dir)


if __name__ == "__main__":
    main()