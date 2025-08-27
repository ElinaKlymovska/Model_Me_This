#!/usr/bin/env python3
"""
Test detection accuracy and bbox quality across all detectors.
"""

import sys
from pathlib import Path
import yaml
import cv2
import numpy as np
import json
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "adetailer_2cn_plus"))

from ad2cn.config import Config
from ad2cn.pipeline.detect import DetectionPipeline
from ad2cn.postprocess.face_inpainting import FaceInpainting
from ad2cn.utils.io import load_image, save_image


class DetectionAccuracyTester:
    """Test detection accuracy and bbox quality."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.results = {}
        
        # Load configuration
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        self.config = Config(**config_data)
        
        # Initialize pipeline
        self.pipeline = DetectionPipeline(self.config.pipeline.model_dump())
        
        print(f"✓ Initialized pipeline with detectors: {list(self.pipeline.detectors.keys())}")
    
    def validate_bbox_quality(self, bbox: List[int], image_shape: tuple) -> Dict[str, Any]:
        """Validate bbox quality and return metrics."""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Basic dimensions
        face_width = x2 - x1
        face_height = y2 - y1
        face_area = face_width * face_height
        image_area = w * h
        
        # Quality metrics
        metrics = {
            'width': face_width,
            'height': face_height,
            'area': face_area,
            'aspect_ratio': face_width / face_height if face_height > 0 else 0,
            'area_percentage': (face_area / image_area) * 100,
            'in_bounds': 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h,
            'reasonable_size': 32 <= face_width <= w*0.8 and 32 <= face_height <= h*0.8,
            'good_aspect_ratio': 0.6 <= face_width/face_height <= 1.4 if face_height > 0 else False
        }
        
        # Overall quality score
        quality_score = 0
        if metrics['in_bounds']:
            quality_score += 25
        if metrics['reasonable_size']:
            quality_score += 25
        if metrics['good_aspect_ratio']:
            quality_score += 25
        if 5 <= metrics['area_percentage'] <= 60:  # Face should be 5-60% of image
            quality_score += 25
            
        metrics['quality_score'] = quality_score
        metrics['is_good_bbox'] = quality_score >= 75
        
        return metrics
    
    def test_single_image(self, image_path: str) -> Dict[str, Any]:
        """Test detection on single image."""
        print(f"\n🖼️ Testing: {Path(image_path).name}")
        
        # Load image
        try:
            image = load_image(image_path)
            print(f"  Image loaded: {image.shape}")
        except Exception as e:
            return {'error': f'Failed to load image: {e}'}
        
        # Run detection
        try:
            detections = self.pipeline.detect(image)
            print(f"  Detection complete: {len(detections)} faces found")
        except Exception as e:
            return {'error': f'Detection failed: {e}'}
        
        # Analyze results
        result = {
            'image_path': str(image_path),
            'image_shape': image.shape,
            'num_faces': len(detections),
            'detections': [],
            'bbox_quality': {
                'good_bboxes': 0,
                'total_bboxes': len(detections),
                'avg_quality_score': 0
            }
        }
        
        quality_scores = []
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection.get('confidence', 0)
            detector_name = detection.get('detector', 'unknown')
            
            # Validate bbox quality
            bbox_metrics = self.validate_bbox_quality(bbox, image.shape)
            quality_scores.append(bbox_metrics['quality_score'])
            
            detection_result = {
                'id': i,
                'bbox': bbox,
                'confidence': confidence,
                'detector': detector_name,
                'bbox_metrics': bbox_metrics
            }
            
            result['detections'].append(detection_result)
            
            if bbox_metrics['is_good_bbox']:
                result['bbox_quality']['good_bboxes'] += 1
            
            print(f"    Face {i+1}: {detector_name} (conf: {confidence:.3f}, quality: {bbox_metrics['quality_score']}/100)")
        
        # Calculate averages
        if quality_scores:
            result['bbox_quality']['avg_quality_score'] = sum(quality_scores) / len(quality_scores)
            result['bbox_quality']['quality_percentage'] = (result['bbox_quality']['good_bboxes'] / len(detections)) * 100
        
        return result
    
    def test_directory(self, input_dir: str) -> Dict[str, Any]:
        """Test detection on directory of images."""
        print(f"\n📁 Testing directory: {input_dir}")
        
        input_path = Path(input_dir)
        if not input_path.exists():
            return {'error': f'Directory not found: {input_dir}'}
        
        # Find image files
        supported_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_files = [
            f for f in input_path.iterdir() 
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        if not image_files:
            return {'error': 'No image files found'}
        
        print(f"Found {len(image_files)} images to test")
        
        # Test each image
        results = []
        summary_stats = {
            'total_images': len(image_files),
            'successful_detections': 0,
            'failed_detections': 0,
            'total_faces_detected': 0,
            'images_with_faces': 0,
            'bbox_quality_stats': {
                'total_bboxes': 0,
                'good_bboxes': 0,
                'avg_quality_score': 0
            }
        }
        
        quality_scores = []
        
        for image_file in sorted(image_files):
            result = self.test_single_image(str(image_file))
            results.append(result)
            
            if 'error' in result:
                summary_stats['failed_detections'] += 1
                print(f"  ❌ Failed: {result['error']}")
            else:
                summary_stats['successful_detections'] += 1
                num_faces = result['num_faces']
                summary_stats['total_faces_detected'] += num_faces
                
                if num_faces > 0:
                    summary_stats['images_with_faces'] += 1
                    
                    # Bbox quality stats
                    bbox_quality = result['bbox_quality']
                    summary_stats['bbox_quality_stats']['total_bboxes'] += bbox_quality['total_bboxes']
                    summary_stats['bbox_quality_stats']['good_bboxes'] += bbox_quality['good_bboxes']
                    
                    if bbox_quality['avg_quality_score'] > 0:
                        quality_scores.append(bbox_quality['avg_quality_score'])
        
        # Calculate final averages
        if quality_scores:
            summary_stats['bbox_quality_stats']['avg_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        if summary_stats['bbox_quality_stats']['total_bboxes'] > 0:
            summary_stats['bbox_quality_stats']['quality_percentage'] = (
                summary_stats['bbox_quality_stats']['good_bboxes'] / 
                summary_stats['bbox_quality_stats']['total_bboxes']
            ) * 100
        else:
            summary_stats['bbox_quality_stats']['quality_percentage'] = 0
        
        # Detection success rate
        summary_stats['face_detection_rate'] = (summary_stats['images_with_faces'] / summary_stats['total_images']) * 100
        
        return {
            'directory': str(input_dir),
            'summary': summary_stats,
            'detailed_results': results
        }
    
    def create_visual_report(self, test_results: Dict, output_dir: str):
        """Create visual report with bbox overlays."""
        print(f"\n📊 Creating visual report in {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if 'detailed_results' not in test_results:
            return
        
        for result in test_results['detailed_results']:
            if 'error' in result or result['num_faces'] == 0:
                continue
                
            image_path = result['image_path']
            image = load_image(image_path)
            
            # Draw bboxes with quality indicators
            for detection in result['detections']:
                bbox = detection['bbox']
                quality_score = detection['bbox_metrics']['quality_score']
                detector = detection['detector']
                confidence = detection['confidence']
                
                x1, y1, x2, y2 = bbox
                
                # Color based on quality: green = good, yellow = ok, red = bad
                if quality_score >= 75:
                    color = (0, 255, 0)  # Green
                elif quality_score >= 50:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Add labels
                label = f"{detector} ({confidence:.2f}) Q:{quality_score}/100"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                cv2.rectangle(image, 
                             (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), 
                             color, -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save annotated image
            output_file = output_path / f"tested_{Path(image_path).stem}.jpg"
            save_image(image, str(output_file))
        
        print(f"✓ Visual report saved to {output_dir}")
    
    def save_json_report(self, test_results: Dict, output_file: str):
        """Save detailed results as JSON."""
        with open(output_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        print(f"✓ JSON report saved to {output_file}")
    
    def print_summary_report(self, test_results: Dict):
        """Print summary report to console."""
        if 'summary' not in test_results:
            return
            
        summary = test_results['summary']
        
        print("\n" + "="*60)
        print("📊 DETECTION ACCURACY SUMMARY")
        print("="*60)
        
        print(f"📁 Directory: {test_results.get('directory', 'N/A')}")
        print(f"🖼️  Total images: {summary['total_images']}")
        print(f"✅ Successful detections: {summary['successful_detections']}")
        print(f"❌ Failed detections: {summary['failed_detections']}")
        print(f"👤 Total faces detected: {summary['total_faces_detected']}")
        print(f"🎯 Images with faces: {summary['images_with_faces']} ({summary['face_detection_rate']:.1f}%)")
        
        print("\n📏 BBOX QUALITY ANALYSIS")
        bbox_stats = summary['bbox_quality_stats']
        print(f"📦 Total bboxes: {bbox_stats['total_bboxes']}")
        print(f"✅ Good quality bboxes: {bbox_stats['good_bboxes']}")
        print(f"📊 Quality percentage: {bbox_stats.get('quality_percentage', 0):.1f}%")
        print(f"⭐ Average quality score: {bbox_stats['avg_quality_score']:.1f}/100")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        if summary['face_detection_rate'] < 80:
            print("⚠️  Low face detection rate - consider tuning detector parameters")
        if bbox_stats.get('quality_percentage', 0) < 70:
            print("⚠️  Poor bbox quality - enable MediaPipe for better accuracy")
        if bbox_stats['avg_quality_score'] < 60:
            print("⚠️  Very low bbox quality - check image quality and detector settings")
            
        print("="*60)


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python3 test_detection_accuracy.py <input_dir> [output_dir]")
        print("Example: python3 test_detection_accuracy.py data/samples/Massy output/test_results")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/detection_test_results"
    config_path = "adetailer_2cn_plus/config.yaml"
    
    print("🧪 DETECTION ACCURACY TESTER")
    print("="*40)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Config: {config_path}")
    
    # Initialize tester
    tester = DetectionAccuracyTester(config_path)
    
    # Run tests
    results = tester.test_directory(input_dir)
    
    if 'error' in results:
        print(f"❌ Test failed: {results['error']}")
        sys.exit(1)
    
    # Generate reports
    tester.print_summary_report(results)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save reports
    tester.save_json_report(results, f"{output_dir}/detection_report.json")
    tester.create_visual_report(results, f"{output_dir}/visual_report")
    
    print(f"\n✅ Testing complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()