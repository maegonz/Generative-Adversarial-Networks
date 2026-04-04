import torch
import torch.nn as nn

## Convolutional Block for GAN's Generator network
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, out_padding=0, bias=True, **kwargs):
        """
        Initialize a convolutional transpose block for a generator network.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        bias : bool, optional
            Whether to use bias in the convolution layer. Default is False.
        **kwargs
            Additional keyword arguments to pass to the convolution layer
        """
        super(GenBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=out_padding, bias=bias, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        # self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.conv(x)
        output = self.norm(output)
        output = self.relu(output)
        return output

    
## Convolutional block used in the Discriminator network
class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True, **kwargs):
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
        padding : int, optional
            Padding for the convolution. Default is 1.
        **kwargs
            Additional keyword arguments to pass to the convolution layer
        """
        super(DiscBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, **kwargs)
        # Normalize features for better training stability
        # self.norm = nn.BatchNorm2d(out_channels)
        # self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        # Leaky ReLU activation to allow small gradients for negative inputs
        self.lk_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        output = self.conv(x)
        # output = self.norm(output)
        output = self.lk_relu(output)
        return output