import torch.nn as nn
from .blocks import DiscBlock

## PatchGAN Discriminator network for real vs fake image patches classification
class Discriminator(nn.Module):
    def __init__(self, in_channels, num_features=64):
        """
        Initialize the Discriminator network.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels
        num_features : int, optional
            Number of features in the first convolutional layer. Default is 64.
        """
        super(Discriminator, self).__init__()
        # Initial convolution block without normalization (as per PatchGAN architecture)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Downsampling layers to reduce spatial dimensions and increase feature depth
        self.downsampling = nn.Sequential(
            DiscBlock(num_features, num_features * 2, stride=2),
            DiscBlock(num_features * 2, num_features * 4, stride=2),
            DiscBlock(num_features * 4, num_features * 8, stride=1)
        )

        # Final convolution to produce a single-channel output (real/fake prediction)
        self.final = nn.Conv2d(num_features * 8, 1, kernel_size=4, stride=1, padding=1)

        # Output activation to map values between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.initial(x)
        output = self.downsampling(output)
        output = self.final(output)
        output = self.sigmoid(output)
        return output