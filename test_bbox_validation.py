#!/usr/bin/env python3
"""
Test bbox validation and refinement functionality.
"""

import sys
from pathlib import Path
import yaml
import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "adetailer_2cn_plus"))

from ad2cn.config import Config
from ad2cn.postprocess.facial_contouring import ContouringMask
from ad2cn.utils.io import load_image, save_image


class BboxValidationTester:
    """Test bbox validation and refinement."""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        self.config = Config(**config_data)
        
        # Initialize contouring system for bbox validation
        self.contouring = ContouringMask(self.config.facial_enhancement.model_dump())
        
    def create_test_bboxes(self, image_shape: tuple) -> list:
        """Create various test bboxes for validation."""
        h, w = image_shape[:2]
        
        test_cases = [
            # Good bbox
            ([w//4, h//4, 3*w//4, 3*h//4], "Normal face bbox"),
            
            # Too large bbox
            ([10, 10, w-10, h-10], "Too large bbox"),
            
            # Too small bbox
            ([w//2-10, h//2-10, w//2+10, h//2+10], "Too small bbox"), 
            
            # Out of bounds bbox
            ([-50, -50, w//2, h//2], "Out of bounds bbox"),
            
            # Invalid bbox (x2 < x1)
            ([w//2, h//4, w//4, 3*h//4], "Invalid bbox (x2 < x1)"),
            
            # Bad aspect ratio (too wide)
            ([w//4, h//2-20, 3*w//4, h//2+20], "Bad aspect ratio (too wide)"),
            
            # Bad aspect ratio (too tall)  
            ([w//2-30, h//4, w//2+30, 3*h//4], "Bad aspect ratio (too tall)"),
            
            # Edge case - at image boundary
            ([0, 0, w//3, h//3], "At image boundary"),
            
            # Reasonable portrait bbox
            ([w//3, h//6, 2*w//3, 2*h//3], "Portrait-style bbox"),
        ]
        
        return test_cases
    
    def test_bbox_validation(self, image_path: str, output_dir: str):
        """Test bbox validation on sample image."""
        print(f"\n🔍 Testing bbox validation on: {Path(image_path).name}")
        
        # Load image
        image = load_image(image_path)
        h, w = image.shape[:2]
        print(f"Image shape: {image.shape}")
        
        # Create test bboxes
        test_bboxes = self.create_test_bboxes(image.shape)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, (bbox, description) in enumerate(test_bboxes):
            print(f"\n  Test {i+1}: {description}")
            print(f"    Original bbox: {bbox}")
            
            # Test validation
            is_valid = self.contouring._validate_bbox(bbox, image.shape)
            print(f"    Is valid: {is_valid}")
            
            # Test refinement
            try:
                refined_bbox = self.contouring._validate_and_fix_bbox(bbox, image.shape)
                print(f"    Refined bbox: {refined_bbox}")
                
                # Test with MediaPipe-style landmarks (simulate)
                fake_landmarks = self._create_fake_landmarks(refined_bbox)
                mp_refined_bbox = self.contouring._refine_face_bbox(
                    bbox, fake_landmarks, image.shape
                )
                print(f"    MP refined bbox: {mp_refined_bbox}")
                
            except Exception as e:
                print(f"    Refinement error: {e}")
                refined_bbox = bbox
                mp_refined_bbox = bbox
            
            # Create visualization
            vis_image = image.copy()
            
            # Draw original bbox in red
            if self._is_drawable_bbox(bbox, image.shape):
                cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            (0, 0, 255), 2)
                cv2.putText(vis_image, "Original", (bbox[0], bbox[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw refined bbox in green
            if refined_bbox and self._is_drawable_bbox(refined_bbox, image.shape):
                cv2.rectangle(vis_image, (refined_bbox[0], refined_bbox[1]), 
                            (refined_bbox[2], refined_bbox[3]), (0, 255, 0), 2)
                cv2.putText(vis_image, "Refined", (refined_bbox[0], refined_bbox[1]-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw MP refined bbox in blue
            if mp_refined_bbox and self._is_drawable_bbox(mp_refined_bbox, image.shape):
                cv2.rectangle(vis_image, (mp_refined_bbox[0], mp_refined_bbox[1]), 
                            (mp_refined_bbox[2], mp_refined_bbox[3]), (255, 0, 0), 2)
                cv2.putText(vis_image, "MP Refined", (mp_refined_bbox[0], mp_refined_bbox[1]-50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Add title
            cv2.putText(vis_image, f"Test {i+1}: {description}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Save visualization
            output_file = Path(output_dir) / f"bbox_test_{i+1:02d}_{description.lower().replace(' ', '_')}.jpg"
            save_image(vis_image, str(output_file))
            
            # Store results
            results.append({
                'test_id': i+1,
                'description': description,
                'original_bbox': bbox,
                'is_valid': is_valid,
                'refined_bbox': refined_bbox,
                'mp_refined_bbox': mp_refined_bbox,
                'output_file': str(output_file)
            })
        
        return results
    
    def _is_drawable_bbox(self, bbox: list, image_shape: tuple) -> bool:
        """Check if bbox can be drawn on image."""
        if not bbox or len(bbox) != 4:
            return False
        
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Check if bbox has positive dimensions and is within reasonable bounds
        return (x2 > x1 and y2 > y1 and 
                -w < x1 < 2*w and -h < y1 < 2*h and
                -w < x2 < 2*w and -h < y2 < 2*h)
    
    def _create_fake_landmarks(self, bbox: list) -> dict:
        """Create fake MediaPipe-style landmarks for testing."""
        if not bbox or len(bbox) != 4:
            return None
            
        x1, y1, x2, y2 = bbox
        face_width = x2 - x1
        face_height = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Create face oval points (simplified)
        face_oval_points = []
        num_points = 16
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            # Create elliptical shape
            radius_x = face_width * 0.4
            radius_y = face_height * 0.45
            
            x = int(center_x + radius_x * np.cos(angle))
            y = int(center_y + radius_y * np.sin(angle))
            face_oval_points.append([x, y])
        
        return {
            'face_oval': face_oval_points,
            'left_eye': [[center_x - face_width//6, center_y - face_height//6]],
            'right_eye': [[center_x + face_width//6, center_y - face_height//6]],
            'nose': [[center_x, center_y]],
            'mouth': [[center_x, center_y + face_height//4]]
        }
    
    def create_summary_report(self, results: list, output_dir: str):
        """Create summary report of bbox validation tests."""
        report_path = Path(output_dir) / "bbox_validation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("BBOX VALIDATION TEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            total_tests = len(results)
            valid_original = sum(1 for r in results if r['is_valid'])
            successful_refinements = sum(1 for r in results if r['refined_bbox'] != r['original_bbox'])
            
            f.write(f"Total tests: {total_tests}\n")
            f.write(f"Valid original bboxes: {valid_original}/{total_tests} ({valid_original/total_tests*100:.1f}%)\n")
            f.write(f"Successful refinements: {successful_refinements}/{total_tests} ({successful_refinements/total_tests*100:.1f}%)\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            for result in results:
                f.write(f"\nTest {result['test_id']}: {result['description']}\n")
                f.write(f"  Original bbox: {result['original_bbox']}\n")
                f.write(f"  Is valid: {result['is_valid']}\n")
                f.write(f"  Refined bbox: {result['refined_bbox']}\n")
                f.write(f"  MP refined bbox: {result['mp_refined_bbox']}\n")
                f.write(f"  Visualization: {result['output_file']}\n")
        
        print(f"✓ Summary report saved to: {report_path}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python3 test_bbox_validation.py <image_path> [output_dir]")
        print("Example: python3 test_bbox_validation.py data/samples/Massy/sample.webp output/bbox_test")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/bbox_validation_test"
    config_path = "adetailer_2cn_plus/config.yaml"
    
    print("🔧 BBOX VALIDATION TESTER")
    print("=" * 40)
    print(f"Image: {image_path}")
    print(f"Output: {output_dir}")
    print(f"Config: {config_path}")
    
    # Initialize tester
    tester = BboxValidationTester(config_path)
    
    # Run tests
    results = tester.test_bbox_validation(image_path, output_dir)
    
    # Create summary report
    tester.create_summary_report(results, output_dir)
    
    print(f"\n✅ Bbox validation testing complete!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"🖼️  Check visualizations: {output_dir}/*.jpg")
    print(f"📄 Summary report: {output_dir}/bbox_validation_report.txt")


if __name__ == "__main__":
    main()