"""
Modular, Composable Image Preprocessing Pipeline.

This module provides production-ready image preprocessing functions designed for
traditional ML models (e.g., SVM, Random Forest). Functions are highly modular,
composable, and can be chained together in various configurations.

Features:
    - Flatten images to 1D vectors for ML model input
    - Normalization (pixel value scaling)
    - Image resizing with aspect ratio preservation
    - Grayscale conversion
    - Noise reduction (bilateral filtering)
    - Pipeline class for composable function chaining

Example Usage:
    >>> from image_preprocessing import ImagePipeline
    >>> pipeline = ImagePipeline([
    ...     ('grayscale', {'force': True}),
    ...     ('resize', {'target_size': (64, 64)}),
    ...     ('normalize', {'method': 'minmax'}),
    ...     ('flatten', {})
    ... ])
    >>> features = pipeline.process(image_array)

Requirements:
    - numpy
    - opencv-python (cv2)
    - scikit-image
    - Pillow
"""

from typing import Callable, List, Tuple, Dict, Any, Union, Optional
import numpy as np
import cv2
from skimage import exposure
from PIL import Image as PILImage


# ============================================================================
# CORE FUNCTION: Flatten/Vectorize for ML Models
# ============================================================================

def flatten_image(
    image_array: np.ndarray,
    preserve_structure: bool = False
) -> np.ndarray:
    """
    Flatten a 2D/3D image array into a 1D feature vector for traditional ML models.
    
    Converts images (grayscale or color) into flat vectors suitable for SVM,
    Random Forest, or other vectorized ML algorithms. Supports optional 
    preservation of channel structure for multi-channel images.
    
    Args:
        image_array: Input image as np.ndarray (2D for grayscale, 3D for color).
                     Expected shape: (height, width) or (height, width, channels).
        preserve_structure: If True, concatenates channels separately to preserve
                          spatial channel information. If False, flattens entirely. 
                          Default: False.
    
    Returns:
        Flattened 1D array of dtype float32.
    
    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If image has unsupported number of dimensions (not 2D or 3D).
    
    Example:
        >>> image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>> vector = flatten_image(image)
        >>> vector.shape
        (150528,)
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image_array)}")
    
    if image_array.ndim not in (2, 3):
        raise ValueError(
            f"Image must be 2D (grayscale) or 3D (color). Got shape {image_array.shape}"
        )
    
    image_array = image_array.astype(np.float32)
    
    if image_array.ndim == 2:
        # Grayscale: simple flatten
        return image_array.flatten()
    else:
        # Color image with channels
        if preserve_structure:
            # Flatten each channel and concatenate
            return np.concatenate([channel.flatten() for channel in image_array])
        else:
            # Flatten entire array
            return image_array.flatten()


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def normalize_image(
    image_array: np.ndarray,
    method: str = 'minmax',
    value_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Normalize pixel values to a specified range.
    
    Applies various normalization techniques to scale pixel intensities,
    improving model convergence and performance.
    
    Args:
        image_array: Input image as np.ndarray (uint8 or float).
        method: Normalization method. Options:
            - 'minmax': Scale to [value_range[0], value_range[1]] (default)
            - 'standard': Zero-mean, unit-variance standardization
            - 'histogram': Histogram equalization (grayscale only)
        value_range: Target range for minmax normalization. Default: (0.0, 1.0).
    
    Returns:
        Normalized array of dtype float32.
    
    Raises:
        ValueError: If method is unsupported or histogram equalization used on color image.
    
    Example:
        >>> image = np.array([[50, 100], [150, 200]], dtype=np.uint8)
        >>> normalized = normalize_image(image, method='minmax')
        >>> normalized.min(), normalized.max()
        (0.0, 1.0)
    """
    if method not in ('minmax', 'standard', 'histogram'):
        raise ValueError(f"Unsupported method: {method}. Choose from minmax, standard, histogram.")
    
    image_array = image_array.astype(np.float32)
    
    if method == 'minmax':
        min_val = image_array.min()
        max_val = image_array.max()
        
        if max_val == min_val:
            return np.full_like(image_array, (value_range[0] + value_range[1]) / 2)
        
        normalized = (image_array - min_val) / (max_val - min_val)
        return normalized * (value_range[1] - value_range[0]) + value_range[0]
    
    elif method == 'standard':
        mean = image_array.mean()
        std = image_array.std()
        
        if std == 0:
            return np.zeros_like(image_array)
        
        return (image_array - mean) / std
    
    elif method == 'histogram':
        if image_array.ndim == 3:
            raise ValueError("Histogram equalization requires grayscale image (2D).")
        
        # Convert to uint8 for histogram equalization
        uint8_image = np.clip(image_array, 0, 255).astype(np.uint8)
        equalized = cv2.equalizeHist(uint8_image)
        return equalized.astype(np.float32) / 255.0


def resize_image(
    image_array: np.ndarray,
    target_size: Tuple[int, int],
    preserve_aspect: bool = True,
    interpolation: str = 'bilinear'
) -> np.ndarray:
    """
    Resize image to target dimensions.
    
    Supports aspect-ratio-preserving resizing with optional padding,
    multiple interpolation methods, and works with both grayscale and color images.
    
    Args:
        image_array: Input image as np.ndarray (2D or 3D).
        target_size: Target size as (height, width).
        preserve_aspect: If True, preserves aspect ratio and pads to target_size.
                        If False, stretches to exact target_size. Default: True.
        interpolation: Interpolation method. Options:
            - 'bilinear': Bilinear interpolation (default, good balance)
            - 'nearest': Nearest neighbor (fast, blocky)
            - 'bicubic': Bicubic interpolation (slower, smoother)
            - 'lanczos': Lanczos (high-quality, slowest)
    
    Returns:
        Resized array of same dtype as input.
    
    Raises:
        ValueError: If target_size is invalid or interpolation method is unsupported.
    
    Example:
        >>> image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        >>> resized = resize_image(image, (224, 224), preserve_aspect=True)
        >>> resized.shape[0], resized.shape[1]
        (224, 224)
    """
    if len(target_size) != 2 or any(s <= 0 for s in target_size):
        raise ValueError(f"target_size must be (height, width) with positive values. Got {target_size}")
    
    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    if interpolation not in interp_map:
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    
    interp_flag = interp_map[interpolation]
    target_h, target_w = target_size
    
    if not preserve_aspect:
        return cv2.resize(image_array, (target_w, target_h), interpolation=interp_flag)
    
    # Preserve aspect ratio with padding
    h, w = image_array.shape[:2]
    aspect_ratio = w / h
    target_aspect = target_w / target_h
    
    if aspect_ratio > target_aspect:
        # Image is wider: fit to width
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        # Image is taller: fit to height
        new_h = target_h
        new_w = int(target_h * aspect_ratio)
    
    resized = cv2.resize(image_array, (new_w, new_h), interpolation=interp_flag)
    
    # Pad to target size
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    pad_h_extra = target_h - new_h - pad_h
    pad_w_extra = target_w - new_w - pad_w
    
    if image_array.ndim == 2:
        padded = np.pad(
            resized,
            ((pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)),
            mode='constant',
            constant_values=0
        )
    else:
        padded = np.pad(
            resized,
            ((pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra), (0, 0)),
            mode='constant',
            constant_values=0
        )
    
    return padded


def to_grayscale(
    image_array: np.ndarray,
    force: bool = False
) -> np.ndarray:
    """
    Convert image to grayscale.
    
    Converts color (RGB/BGR) images to single-channel grayscale using standard
    weights. Grayscale images are returned unchanged unless force=True.
    
    Args:
        image_array: Input image as np.ndarray (2D for grayscale, 3D for color).
        force: If True, applies grayscale conversion even if already grayscale.
               Default: False.
    
    Returns:
        Grayscale array of shape (height, width) and dtype same as input.
    
    Raises:
        ValueError: If image has unexpected number of dimensions.
    
    Example:
        >>> color_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>> gray = to_grayscale(color_image)
        >>> gray.shape
        (224, 224)
    """
    if image_array.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {image_array.shape}")
    
    if image_array.ndim == 2:
        if force:
            # Apply additional processing even if already grayscale
            return image_array.astype(image_array.dtype)
        return image_array
    
    # Convert 3D color to 2D grayscale using standard weights
    # Using luminance formula: 0.299*R + 0.587*G + 0.114*B
    grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return grayscale.astype(image_array.dtype)


def reduce_noise(
    image_array: np.ndarray,
    method: str = 'bilateral',
    kernel_size: int = 5,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Reduce noise in image using various filtering techniques.
    
    Applies filtering to suppress noise while preserving edges and important features.
    Multiple methods available for different noise characteristics.
    
    Args:
        image_array: Input image as np.ndarray (uint8 preferred for bilateral).
        method: Noise reduction method. Options:
            - 'bilateral': Bilateral filtering (edge-preserving, default)
            - 'gaussian': Gaussian blur (simple but blurs edges)
            - 'morphological': Morphological opening+closing (binary-like noise)
            - 'median': Median filtering (good for salt-and-pepper noise)
        kernel_size: Kernel size for filtering (must be odd). Default: 5.
        sigma_color: Color sigma for bilateral filter. Default: 75.0.
        sigma_space: Spatial sigma for bilateral filter. Default: 75.0.
    
    Returns:
        Filtered array of same shape and dtype as input.
    
    Raises:
        ValueError: If method is unsupported or kernel_size is even/invalid.
    
    Example:
        >>> noisy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>> denoised = reduce_noise(noisy_image, method='bilateral')
        >>> denoised.shape == noisy_image.shape
        True
    """
    if method not in ('bilateral', 'gaussian', 'morphological', 'median'):
        raise ValueError(f"Unsupported method: {method}")
    
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive and odd. Got {kernel_size}")
    
    if method == 'bilateral':
        if image_array.ndim == 2:
            return cv2.bilateralFilter(
                image_array.astype(np.uint8),
                kernel_size,
                sigma_color,
                sigma_space
            )
        else:
            return cv2.bilateralFilter(
                image_array.astype(np.uint8),
                kernel_size,
                sigma_color,
                sigma_space
            )
    
    elif method == 'gaussian':
        return cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)
    
    elif method == 'morphological':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(image_array.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    elif method == 'median':
        return cv2.medianBlur(image_array.astype(np.uint8), kernel_size)


# ============================================================================
# PIPELINE CLASS: Composable Function Chaining
# ============================================================================

class ImagePipeline:
    """
    Composable image preprocessing pipeline.
    
    Chains multiple preprocessing operations in sequence, allowing flexible
    configuration and reuse. Each operation is specified as a tuple of
    (function_name, kwargs).
    
    Supported operations:
        - 'flatten': Flatten image to 1D vector (flatten_image)
        - 'normalize': Normalize pixel values (normalize_image)
        - 'resize': Resize image (resize_image)
        - 'grayscale': Convert to grayscale (to_grayscale)
        - 'denoise': Reduce noise (reduce_noise)
    
    Example:
        >>> pipeline = ImagePipeline([
        ...     ('grayscale', {}),
        ...     ('resize', {'target_size': (64, 64)}),
        ...     ('normalize', {'method': 'minmax'}),
        ...     ('flatten', {})
        ... ])
        >>> image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>> features = pipeline.process(image)
    """
    
    # Map operation names to functions
    OPERATIONS: Dict[str, Callable] = {
        'flatten': flatten_image,
        'normalize': normalize_image,
        'resize': resize_image,
        'grayscale': to_grayscale,
        'denoise': reduce_noise,
    }
    
    def __init__(self, operations: List[Tuple[str, Dict[str, Any]]]):
        """
        Initialize pipeline with ordered operations.
        
        Args:
            operations: List of (operation_name, kwargs) tuples.
                       Example: [('grayscale', {}), ('resize', {'target_size': (64, 64)})]
        
        Raises:
            ValueError: If any operation name is not supported.
        """
        self.operations = operations
        
        for op_name, _ in operations:
            if op_name not in self.OPERATIONS:
                raise ValueError(
                    f"Unknown operation: {op_name}. "
                    f"Supported: {', '.join(self.OPERATIONS.keys())}"
                )
    
    def process(self, image_array: np.ndarray) -> np.ndarray:
        """
        Apply all pipeline operations in sequence to image.
        
        Args:
            image_array: Input image as np.ndarray.
        
        Returns:
            Processed image/vector after all operations.
        
        Raises:
            Exception: If any operation fails (with context about which operation).
        """
        result = image_array.copy()
        
        for op_name, kwargs in self.operations:
            try:
                operation = self.OPERATIONS[op_name]
                result = operation(result, **kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"Pipeline failed at operation '{op_name}' with kwargs {kwargs}. "
                    f"Error: {str(e)}"
                ) from e
        
        return result
    
    def add_operation(self, operation_name: str, kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an operation to the end of the pipeline.
        
        Args:
            operation_name: Name of operation to add.
            kwargs: Operation parameters. Default: {}.
        
        Raises:
            ValueError: If operation name is not supported.
        """
        if operation_name not in self.OPERATIONS:
            raise ValueError(f"Unknown operation: {operation_name}")
        
        self.operations.append((operation_name, kwargs or {}))
    
    def __repr__(self) -> str:
        """Return string representation of pipeline."""
        ops_str = " -> ".join(f"{name}({', '.join(f'{k}={v}' for k, v in kwargs.items())})" 
                              for name, kwargs in self.operations)
        return f"ImagePipeline([{ops_str}])"


# ============================================================================
# FUNCTIONAL COMPOSITION: Decorator-based Chaining
# ============================================================================

def compose(*functions: Callable) -> Callable:
    """
    Compose functions right-to-left (mathematical composition).
    
    Creates a single function that applies provided functions in sequence,
    with output of one function fed to the next.
    
    Args:
        *functions: Variable number of callable functions.
    
    Returns:
        Composed function that applies all operations in sequence.
    
    Example:
        >>> from functools import partial
        >>> resize_64 = partial(resize_image, target_size=(64, 64))
        >>> normalize_minmax = partial(normalize_image, method='minmax')
        >>> flatten = flatten_image
        >>> pipeline = compose(flatten, normalize_minmax, resize_64)
        >>> image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>> features = pipeline(image)
    """
    def composed(arg: Any) -> Any:
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    
    return composed


def pipeline_decorator(*operations: Tuple[Callable, Dict[str, Any]]) -> Callable:
    """
    Decorator to create preprocessing pipeline from decorated function.
    
    Allows decorating a custom function with preprocessing operations
    that are automatically applied before the function receives the image.
    
    Args:
        *operations: Variable number of (function, kwargs) tuples.
    
    Returns:
        Decorator that wraps a function with preprocessing.
    
    Example:
        >>> @pipeline_decorator(
        ...     (to_grayscale, {}),
        ...     (lambda x: resize_image(x, (64, 64)), {}),
        ...     (flatten_image, {})
        ... )
        >>> def extract_features(image):
        ...     return image  # Already preprocessed
        >>> image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>> features = extract_features(image)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(image_array: np.ndarray) -> Any:
            result = image_array.copy()
            for operation, kwargs in operations:
                result = operation(result, **kwargs)
            return func(result)
        return wrapper
    
    return decorator


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load image from raw bytes using PIL.
    
    Args:
        image_bytes: Raw byte content of image file.
    
    Returns:
        Image as np.ndarray (RGB format).
    
    Raises:
        ValueError: If bytes cannot be read as image.
    """
    import io
    try:
        pil_image = PILImage.open(io.BytesIO(image_bytes))
        return np.array(pil_image)
    except Exception as e:
        raise ValueError(f"Failed to load image from bytes: {str(e)}")


def load_image_from_file(file_path: str) -> np.ndarray:
    """
    Load image from file using OpenCV.
    
    Args:
        file_path: Path to image file.
    
    Returns:
        Image as np.ndarray (BGR format from OpenCV).
    
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file cannot be read as image.
    """
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Failed to read image: {file_path}")
    return image


def load_image_from_pil(pil_image: PILImage.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.
    
    Args:
        pil_image: PIL Image object.
    
    Returns:
        Image as np.ndarray.
    """
    return np.array(pil_image)


def batch_process(
    images: List[np.ndarray],
    pipeline: ImagePipeline
) -> np.ndarray:
    """
    Apply pipeline to batch of images.
    
    Args:
        images: List of image arrays.
        pipeline: ImagePipeline instance.
    
    Returns:
        Array of shape (batch_size, *processed_shape).
    """
    processed = [pipeline.process(img) for img in images]
    
    # Stack arrays (all must have same shape)
    return np.array(processed)


if __name__ == "__main__":
    # Example usage demonstrating all functions
    print("=" * 70)
    print("Image Preprocessing Pipeline - Example Usage")
    print("=" * 70)
    
    # Create sample image
    sample_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    print(f"\n1. Original image shape: {sample_image.shape}")
    
    # Individual function examples
    print("\n2. Individual Function Examples:")
    
    gray = to_grayscale(sample_image)
    print(f"   - Grayscale: {gray.shape}")
    
    resized = resize_image(sample_image, (64, 64), preserve_aspect=True)
    print(f"   - Resized: {resized.shape}")
    
    normalized = normalize_image(sample_image, method='minmax')
    print(f"   - Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    denoised = reduce_noise(sample_image, method='bilateral')
    print(f"   - Denoised: {denoised.shape}")
    
    flattened = flatten_image(sample_image)
    print(f"   - Flattened: {flattened.shape}")
    
    # Pipeline class example
    print("\n3. Pipeline Class Example:")
    pipeline = ImagePipeline([
        ('grayscale', {}),
        ('resize', {'target_size': (64, 64), 'preserve_aspect': True}),
        ('denoise', {'method': 'bilateral', 'kernel_size': 5}),
        ('normalize', {'method': 'minmax'}),
        ('flatten', {})
    ])
    print(f"   Pipeline: {pipeline}")
    
    features = pipeline.process(sample_image)
    print(f"   Output shape: {features.shape}")
    print(f"   Output dtype: {features.dtype}")
    print(f"   Value range: [{features.min():.3f}, {features.max():.3f}]")
    
    # Functional composition example
    print("\n4. Functional Composition Example:")
    from functools import partial
    
    gray_op = partial(to_grayscale)
    resize_op = partial(resize_image, target_size=(64, 64))
    norm_op = partial(normalize_image, method='minmax')
    flat_op = flatten_image
    
    composed_pipeline = compose(flat_op, norm_op, resize_op, gray_op)
    features_composed = composed_pipeline(sample_image)
    print(f"   Composed pipeline output: {features_composed.shape}")
    
    # Batch processing example
    print("\n5. Batch Processing Example:")
    batch_images = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(4)]
    batch_output = batch_process(batch_images, pipeline)
    print(f"   Batch input: {len(batch_images)} images of shape {batch_images[0].shape}")
    print(f"   Batch output shape: {batch_output.shape}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
