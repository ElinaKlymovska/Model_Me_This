"""
Image I/O utilities.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, List
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image as numpy array (H, W, C)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Try OpenCV first (faster for most formats)
        image = cv2.imread(str(image_path))
        if image is not None:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        
        # Fallback to PIL
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
        
        # Convert RGBA to RGB if necessary
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        return image
        
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def save_image(image: np.ndarray, output_path: Union[str, Path], 
               quality: int = 95) -> None:
    """Save image to file.
    
    Args:
        image: Image to save as numpy array
        output_path: Output file path
        quality: JPEG quality (1-100) for JPEG files
        
    Raises:
        ValueError: If image cannot be saved
    """
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Determine format from extension
        ext = output_path.suffix.lower()
        
        if ext in ['.jpg', '.jpeg']:
            # Save as JPEG using PIL for better quality control
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB to BGR for OpenCV
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        elif ext == '.png':
            # Save as PNG
            if len(image.shape) == 3 and image.shape[2] == 3:
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bgr_image)
            else:
                cv2.imwrite(str(output_path), image)
        
        else:
            # Default to PNG
            if len(image.shape) == 3 and image.shape[2] == 3:
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bgr_image)
            else:
                cv2.imwrite(str(output_path), image)
                
    except Exception as e:
        raise ValueError(f"Failed to save image to {output_path}: {e}")


def load_images_from_directory(directory: Union[str, Path], 
                             extensions: Optional[List[str]] = None) -> List[np.ndarray]:
    """Load all images from a directory.
    
    Args:
        directory: Directory path
        extensions: List of file extensions to include (e.g., ['.jpg', '.png'])
        
    Returns:
        List of loaded images
    """
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    images = []
    image_files = []
    
    # Collect image files
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    # Sort files for consistent ordering
    image_files.sort()
    
    # Load images
    for image_file in image_files:
        try:
            image = load_image(image_file)
            images.append(image)
            logger.info(f"Loaded image: {image_file}")
        except Exception as e:
            logger.warning(f"Failed to load {image_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(images)} images from {directory}")
    return images


def resize_image(image: np.ndarray, target_size: tuple, 
                interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target size as (width, height)
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    if len(target_size) != 2:
        raise ValueError("target_size must be (width, height)")
    
    width, height = target_size
    
    # OpenCV expects (width, height) for resize
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    
    return resized


def normalize_image(image: np.ndarray, target_range: tuple = (0, 1)) -> np.ndarray:
    """Normalize image to target range.
    
    Args:
        image: Input image
        target_range: Target range as (min, max)
        
    Returns:
        Normalized image
    """
    if len(target_range) != 2:
        raise ValueError("target_range must be (min, max)")
    
    min_val, max_val = target_range
    
    # Convert to float if needed
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Normalize to target range
    normalized = (image - image.min()) / (image.max() - image.min())
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized


def get_image_info(image: np.ndarray) -> dict:
    """Get information about an image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image information
    """
    info = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min_value': float(image.min()),
        'max_value': float(image.max()),
        'mean_value': float(image.mean()),
        'std_value': float(image.std())
    }
    
    if len(image.shape) == 3:
        info['channels'] = image.shape[2]
        info['color_space'] = 'RGB' if image.shape[2] == 3 else 'Multi-channel'
    else:
        info['channels'] = 1
        info['color_space'] = 'Grayscale'
    
    return info
