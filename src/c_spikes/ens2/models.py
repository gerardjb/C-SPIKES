# Neural network architecture definitions modeled on ENS2 UNet implementation

import torch
import torch.nn as nn
from collections import OrderedDict

class UNet(nn.Module):
    """
    U-Net architecture for spike inference.
    Uses 1D convolutions implemented as 2D convolutions with one spatial dimension.
    For now, not implementing the other models from ENS2, but this can be extended.
    
    Parameters:
        in_channels (int): Number of input channels (default: 1).
        out_channels (int): Number of output channels (default: 1).
        init_features (int): Initial number of features in the first layer (default: 9).
        kernel_size (int): Size of the convolutional kernel (default: 3).
        padding (int): Padding for the convolutions (default: 1).
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=9, kernel_size=3, padding=1):
        super(UNet, self).__init__()
        features = init_features
        # Encoder path
        self.encoder1 = UNet._block(in_channels, features, name="enc1", kernel_size=kernel_size, padding=padding)
        self.pool1   = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.encoder2 = UNet._block(features, features * 2, name="enc2", kernel_size=kernel_size, padding=padding)
        self.pool2   = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3", kernel_size=kernel_size, padding=padding)
        self.pool3   = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        # Bottleneck
        self.bottleneck = UNet._block(features * 4, features * 8, name="bottleneck", kernel_size=kernel_size, padding=padding)
        # Decoder path
        self.upconv3  = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=(1, 2), stride=(1, 2))
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3", kernel_size=kernel_size, padding=padding)
        self.upconv2  = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=(1, 2), stride=(1, 2))
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2", kernel_size=kernel_size, padding=padding)
        self.upconv1  = nn.ConvTranspose2d(features * 2, features, kernel_size=(1, 2), stride=(1, 2))
        self.decoder1 = UNet._block(features * 2, features, name="dec1", kernel_size=kernel_size, padding=padding)
        # Final 1x1 convolution to produce output channel
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # Input x shape expected: (batch, length) or (batch, 1, length).
        # Reshape to (batch, 1, 1, length) for 2D conv operations.
        if x.dim() == 2:
            x = x.unsqueeze(1).unsqueeze(1)   # -> (batch, 1, 1, length)
        elif x.dim() == 3:
            x = x.unsqueeze(1)               # -> (batch, 1, 1, length)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        # ReLU on output to ensure non-negative spike rates
        out = torch.relu(self.conv(dec1)).squeeze()
        return out

    @staticmethod
    def _block(in_channels, features, name, kernel_size, padding):
        # Helper to create a convolutional block: Conv -> InstanceNorm -> LeakyReLU (x2)
        return nn.Sequential(OrderedDict([
            (name + "_conv1", nn.Conv2d(in_channels, features, kernel_size=kernel_size, padding=padding, bias=False)),
            (name + "_norm1", nn.InstanceNorm2d(num_features=features)),
            (name + "_relu1", nn.LeakyReLU(0.2, inplace=True)),
            (name + "_conv2", nn.Conv2d(features, features, kernel_size=kernel_size, padding=padding, bias=False)),
            (name + "_norm2", nn.InstanceNorm2d(num_features=features)),
            (name + "_relu2", nn.LeakyReLU(0.2, inplace=True))
        ]))
