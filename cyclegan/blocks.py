import torch
import torch.nn as nn

## Convolutional Block for GANs that supports both
## downsampling (Conv2D) and upsampling (ConvTranspose2D)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_activation=True, **kwargs):
        """
        Initialize a convolutional block for GANs.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        down : bool, optional
            If True, use Conv2d for downsampling; if False, use ConvTranspose2d for upsampling. Default is True.
        use_activation : bool, optional
            If True, apply ReLU activation; if False, use Identity. Default is True.
        **kwargs
            Additional keyword arguments to pass to the convolution layer
        """
        super(ConvBlock, self).__init__()
        if down:
            self.conv = nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if use_activation else nn.Identity()

    def forward(self, x):
        output = self.conv(x)
        output = self.norm(output)
        output = self.relu(output)
        return output

## Residual Block for GANs with two Convolutional layers
## in order to learn residual mappings
class ResBlock(nn.Module):
    def __init__(self, channels):
        """
        Initialize a residual block for GANs.
        
        Parameters
        ----------
        channels : int
            Number of input and output channels
        """
        super(ResBlock, self).__init__()
        self.convblock1 = ConvBlock(channels, channels, kernel_size=3, padding=1)
        self.convblock2 = ConvBlock(channels, channels, use_activation=False, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        output = self.convblock1(x)
        output = self.convblock2(output)
        # Residual connection
        output += res
        return output
    
## Convolutional block used in the Discriminator network
class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        """
        Initialize a convolutional block for the Discriminator.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        stride : int, optional
            Stride for the convolution. Default is 2.
        """
        super(DiscBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=True, padding_mode="reflect")
        # Normalize features for better training stability
        self.norm = nn.InstanceNorm2d(out_channels)
        # Leaky ReLU activation to allow small gradients for negative inputs
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        output = self.conv(x)
        output = self.norm(output)
        output = self.leaky_relu(output)
        return output