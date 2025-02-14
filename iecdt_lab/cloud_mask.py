import numpy as np
from PIL import Image

def generate_mask(tile: np.ndarray, ndsi_threshold: float = 0.3) -> np.ndarray:
    """
    Generate a cloud mask based on the Normalized Difference Snow Index (NDSI).

    Args:
        tile (np.ndarray): The image tile (height, width, channels).
        ndsi_threshold (float): The threshold for cloud detection based on NDSI. Default is 0.3.

    Returns:
        np.ndarray: The generated mask, where 1 represents cloud and 0 represents non-cloud.
    """
    blue_channel = tile[:, :, 0]  # Blue channel (assuming RGB)
    red_channel = tile[:, :, 2]   # Red channel (assuming RGB)

    # Calculate NDSI
    ndsi = (blue_channel - red_channel) / (blue_channel + red_channel + 1e-6)

    # Generate cloud mask based on the NDSI threshold
    mask = (ndsi > ndsi_threshold).astype(np.uint8)

    return mask

def save_mask(mask: np.ndarray, filename: str) -> None:
    """
    Save the mask as a PNG image.

    Args:
        mask (np.ndarray): The generated mask.
        filename (str): The filename to save the mask as.
    """
    mask_image = Image.fromarray(mask * 255)  # Convert to 0/255 range for image
    mask_image.save(filename)