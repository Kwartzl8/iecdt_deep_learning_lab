import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, batchnorm: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.batchnorm = nn.BatchNorm2d(out_features) if batchnorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv(x))
        x = self.batchnorm(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int):
        super().__init__()
        self.conv1 = ConvBlock(input_channels, 32, batchnorm=True)  # (B, 32, 128, 128)
        self.conv2 = ConvBlock(32, 64, batchnorm=True)  # (B, 64, 64, 64)
        self.conv3 = ConvBlock(64, 128, batchnorm=True)  # (B, 128, 32, 32)

        # Latent space mapping
        self.flatten_size = 32 * 32 * 128  # Computed from input shape
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)  # Mean layer
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)  # Log variance layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(start_dim=1)  # Flatten for fully connected layers
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 32 * 128)

        self.deconv1 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, input_channels, 3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc(z).view(-1, 128, 32, 32)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        z = torch.sigmoid(self.deconv4(z))  # Ensure output is between [0,1]
        return z


class CNNVAE(nn.Module):
    def __init__(self, latent_dim: int, input_channels: int = 3):
        super().__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(input_channels, latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε * σ, where ε ~ N(0,1)"""
        std = torch.exp(0.5 * logvar)  # Convert log variance to standard deviation
        eps = torch.randn_like(std)  # Sample epsilon from N(0,1)
        return mu + eps * std  # Reparameterized latent variable

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar