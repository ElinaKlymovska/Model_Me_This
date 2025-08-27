"""Simple image I/O utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load image from file path."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
        
    return image


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> bool:
    """Save image to file path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return cv2.imwrite(str(output_path), image)


def load_images_from_directory(directory_path: Union[str, Path]) -> List[np.ndarray]:
    """Load all images from directory."""
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = [p for p in directory_path.iterdir() 
                   if p.is_file() and p.suffix.lower() in supported_extensions]
    
    images = []
    for image_path in sorted(image_paths):
        try:
            image = load_image(image_path)
            images.append(image)
        except Exception as e:
            print(f"Warning: Could not load {image_path}: {e}")
    
    return images