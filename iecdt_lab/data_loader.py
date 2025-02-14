import re
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image  # Make sure you import this if you're saving mask images

@dataclass
class RGBTile:
    time_ix: int
    lat_ix: slice
    lon_ix: slice
    mean_cloud_lengthscale: float
    cloud_fraction: float
    cloud_iorg: float
    fractal_dimension: float

class GOESRGBTiles(Dataset):
    def __init__(self, tiles_file: str, metadata_file: str, transform=None, ndsi_threshold: float = 0.3) -> None:
        """
        Args:
            tiles_file (str): Path to a directory containing all the tiles.
            metadata_file (str): Path to a CSV file containing metadata with the header
                "tile_id,time_ix,lat_ix,lon_ix,$metric1,$metric2,...". `lat_ix` and `lon_ix`
                will be strings in the format "slice(start, stop, None)".
            ndsi_threshold (float): Threshold for cloud mask based on NDSI. Default is 0.3.
        """
        self.tiles_metadata = self._parse_metadata(metadata_file)
        self.tiles_dir = tiles_file
        self.transform = transform
        self.ndsi_threshold = ndsi_threshold  # threshold for cloud mask based on NDSI

    def __len__(self) -> int:
        return len(self.tiles_metadata)

    def __getitem__(self, ix: int) -> tuple[np.ndarray, tuple]:
        tiles_file = f"{self.tiles_dir}/{self.tiles_metadata[ix].time_ix}/time_step.npy"
        tile = np.load(tiles_file)
        tile = tile[
            self.tiles_metadata[ix].lat_ix,
            self.tiles_metadata[ix].lon_ix,
        ]

        # Assuming the tile has the shape (C, H, W) where C is the number of channels
        # Extract the Blue and Red channels (assuming BGR or RGB order, adjust if necessary)
        blue_channel = tile[0]  
        red_channel = tile[2]   
        
        # Calculate NDSI
        ndsi = self.calculate_ndsi(blue_channel, red_channel)

        # Create the cloud mask based on NDSI
        mask = (ndsi > self.ndsi_threshold).astype(np.uint8)  # binary mask based on NDSI threshold

        # Optionally save the mask as a PNG image
        mask_image = Image.fromarray(mask * 255)  # Convert 0/1 mask to 0/255 for image format
        mask_image.save(f"cloud_masks/{self.tiles_metadata[ix].time_ix}_{ix}_mask.png")

        labels = (
            self.tiles_metadata[ix].mean_cloud_lengthscale,
            self.tiles_metadata[ix].cloud_fraction,
            self.tiles_metadata[ix].cloud_iorg,
            self.tiles_metadata[ix].fractal_dimension,
        )
        if self.transform:
            tile = self.transform(tile)

        return tile, labels, mask

    def _generate_cloud_mask(self, tile: np.ndarray) -> np.ndarray:
        # Assuming the tile is in shape (height, width, 3) (RGB)
        blue = tile[:, :, 0]  # Blue channel
        red = tile[:, :, 2]   # Red channel
        
        # Calculate the NDSI (Normalized Difference Snow Index)
        ndsi = (blue - red) / (blue + red)
        
        # Generate the mask based on NDSI threshold (e.g., NDSI > 0.3 indicates clouds)
        mask = (ndsi > 0.3).astype(np.uint8)  # 1 for clouds, 0 for ground
        
        return mask

    def _parse_metadata(self, metadata_file: str) -> list[RGBTile]:
        metadata = pd.read_csv(metadata_file)
        # Parse each row into an RGBTile object
        metadata = [
            RGBTile(
                time_ix=row["time_ix"],
                lat_ix=self._extract_slice(row["lat_ix"]),
                lon_ix=self._extract_slice(row["lon_ix"]),
                mean_cloud_lengthscale=row["mean_cloud_lengthscale"],
                cloud_fraction=row["cloud_fraction"],
                cloud_iorg=row["cloud_iorg"],
                fractal_dimension=row["fractal_dimension"],
            )
            for _, row in metadata.iterrows()
        ]
        return metadata

    @staticmethod
    def _extract_slice(s: str) -> slice:
        """Takes as input a string of the form 'slice($start, $stop, None)' and
        returns a Python slice object."""
        # NOTE: We could use `eval` here but that's not safe.
        match = re.match(r"slice\((\d+), (\d+), None\)", s)
        if match is None:
            raise ValueError(f"Could not match {s}")
        return slice(int(match.group(1)), int(match.group(2)))

    def calculate_ndsi(self, blue_channel: np.ndarray, red_channel: np.ndarray) -> np.ndarray:
        """
        Calculate the Normalized Difference Snow Index (NDSI) for cloud detection
        using the blue and red channels of the image.
        
        Args:
            blue_channel (np.ndarray): The blue channel of the RGB image.
            red_channel (np.ndarray): The red channel of the RGB image.
        
        Returns:
            np.ndarray: The NDSI index calculated from the blue and red channels.
        """
        ndsi = (blue_channel - red_channel) / (blue_channel + red_channel + 1e-6)  # Add small value to avoid division by zero
        return ndsi

def get_data_loaders(
    tiles_path,
    train_metadata,
    val_metadata,
    batch_size,
    data_transforms,
    dataloader_workers,
):
    # Initialize the dataset with the necessary parameters, including the transform and threshold for NDSI.
    train_ds = GOESRGBTiles(
        tiles_file=tiles_path,
        metadata_file=train_metadata,
        transform=data_transforms,
    )
    
    # Create the DataLoader for the training set
    train_data_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
    )
    
    # Initialize the validation dataset
    val_ds = GOESRGBTiles(
        tiles_file=tiles_path,
        metadata_file=val_metadata,
        transform=data_transforms,
    )
    
    # Create the DataLoader for the validation set
    val_data_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,  # Typically, we don't shuffle validation data
        num_workers=dataloader_workers,
    )
    
    return train_data_loader, val_data_loader  # Fix indentation of return statement