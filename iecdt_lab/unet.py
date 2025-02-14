import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    """
    Helper function: Two consecutive 3x3 convolutions with ReLU.
    Padding=1 ensures the spatial dimensions remain unchanged.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = 3):
        super(UNet, self).__init__()
        
        # Encoder (Contracting path)
        self.enc1 = double_conv(input_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = double_conv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = double_conv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = double_conv(128, 256)
        
        # Decoder (Expanding path)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = double_conv(256, 128)  # 128 from upconv + 128 from enc3
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = double_conv(128, 64)   # 64 from upconv + 64 from enc2
        
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = double_conv(64, 32)    # 32 from upconv + 32 from enc1
        
        # Final 1x1 convolution to match output channels
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        enc1 = self.enc1(x)             # [B, 32, 256, 256]
        enc2 = self.enc2(self.pool1(enc1))# [B, 64, 128, 128]
        enc3 = self.enc3(self.pool2(enc2))# [B, 128, 64, 64]
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))  # [B, 256, 32, 32]
        
        # Decoder path
        up1 = self.upconv1(bottleneck)  # Upsample to [B, 128, 64, 64]
        cat1 = torch.cat([up1, enc3], dim=1)  # Skip connection with enc3 → [B, 256, 64, 64]
        dec1 = self.dec1(cat1)          # [B, 128, 64, 64]
        
        up2 = self.upconv2(dec1)        # Upsample to [B, 64, 128, 128]
        cat2 = torch.cat([up2, enc2], dim=1)  # Skip connection with enc2 → [B, 128, 128, 128]
        dec2 = self.dec2(cat2)          # [B, 64, 128, 128]
        
        up3 = self.upconv3(dec2)        # Upsample to [B, 32, 256, 256]
        cat3 = torch.cat([up3, enc1], dim=1)  # Skip connection with enc1 → [B, 64, 256, 256]
        dec3 = self.dec3(cat3)          # [B, 32, 256, 256]
        
        # Final output mapping
        output = self.final_conv(dec3)  # [B, output_channels, 256, 256]
        return output

# Example usage:
if __name__ == '__main__':
    model = UNet(input_channels=3, output_channels=3)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f'Output shape: {output.shape}')  # Expected: [1, 3, 256, 256]