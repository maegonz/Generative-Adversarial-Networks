import torch
import torch.nn as nn
from .blocks import ConvBlock, ResBlock

## Generator network for GANs image-to-image tasks
class Generator(nn.Module):
    def __init__(self, in_channels, num_features=64, num_residuals=9):
        """
        Initialize the Generator network.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels
        num_features : int, optional
            Number of features in the first convolutional layer. Default is 64.
        num_residuals : int, optional
            Number of residual blocks. Default is 9.
        """
        super(Generator, self).__init__()
        # Initial convolution block with 7x7 kernel and reflection padding
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        # Downsampling layers to reduce spatial dimensions
        self.downsampling = nn.Sequential(
            ConvBlock(num_features, num_features * 2, down=True, kernel_size=3, stride=2, padding=1),
            ConvBlock(num_features * 2, num_features * 4, down=True, kernel_size=3, stride=2, padding=1)
        )

        # Residual blocks for learning complex features while preserving input details
        self.residuals = nn.Sequential(
            *[ResBlock(num_features * 4) for _ in range(num_residuals)]
        )

        # Upsampling layers to restore spatial dimensions
        self.upsampling = nn.Sequential(
            ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        # Final convolution to produce the output image with the same number of channels as input
        self.final = nn.Conv2d(num_features, in_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
        
        # Output activation to map values between -1 and 1
        self.tanh = nn.Tanh()

    def forward(self, x):
        output = self.initial(x)
        output = self.downsampling(output)
        output = self.residuals(output)
        output = self.upsampling(output)
        output = self.final(output)
        output = self.tanh(output)
        return output