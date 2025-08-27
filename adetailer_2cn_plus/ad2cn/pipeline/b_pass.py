"""Simple B-Pass enhancement pipeline."""

import numpy as np
from typing import Dict, Any
from pathlib import Path
import cv2


class BPassPipeline:
    """Simple B-Pass enhancement pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get('output_dir', 'output')
        Path(self.output_dir).mkdir(exist_ok=True)
        
    def enhance(self, image: np.ndarray, faces: list) -> np.ndarray:
        """Simple enhancement - just return original image."""
        return image
        
    def process_file(self, input_path: str) -> str:
        """Process single file through B-Pass pipeline."""
        image = cv2.imread(input_path)
        if image is None:
            return None
            
        # Simple processing - copy to output
        output_path = Path(self.output_dir) / f"{Path(input_path).stem}_enhanced.jpg"
        cv2.imwrite(str(output_path), image)
        
        return str(output_path)