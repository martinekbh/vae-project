"""
Code from:
https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py
"""

import torch
from torch import nn, optim
import torch.nn.functional as F

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, n_channels=3):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        #print("ENCODING")
        #print("Size:", x.shape)
        x = torch.relu(self.bn1(self.conv1(x)))
        #print("Size:", x.shape)
        x = self.layer1(x)
        #print("Size:", x.shape)
        x = self.layer2(x)
        #print("Size:", x.shape)
        x = self.layer3(x)
        #print("Size:", x.shape)
        x = self.layer4(x)
        #print("Size:", x.shape)
        x = F.adaptive_avg_pool2d(x, 1)
        #print("Size:", x.shape)
        x = x.view(x.size(0), -1)
        #print("Size:", x.shape)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, img_size=[64, 64],  n_channels=3):
        """
        img_size: [n_channels, height, width] or [height, width]. If given as length 3, the n_channels variable will be ignored.
        n_channels: Number of channels in the input image.
        """
        super().__init__()
        if len(img_size) == 3:
            self.n_channels = img_size[0]
            self.img_size = img_size[1:]
        else: 
            self.n_channels = n_channels
            self.img_size = img_size
        
        self.interpolation_scale = int(img_size[0]/2**4)
        
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, n_channels, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        #print("DECODING")
        x = self.linear(z)
        #print("Size:", x.shape)
        x = x.view(z.size(0), 512, 1, 1)
        #print("Size:", x.shape)
        x = F.interpolate(x, scale_factor=self.interpolation_scale)
        #print("Size:", x.shape, "(after interpolate)")
        x = self.layer4(x)
        #print("Size:", x.shape)
        x = self.layer3(x)
        #print("Size:", x.shape)
        x = self.layer2(x)
        #print("Size:", x.shape)
        x = self.layer1(x)
        #print("Size:", x.shape)
        x = torch.sigmoid(self.conv1(x))
        #x = F.interpolate(x, size=(128,128), mode='bilinear') # Resize if needed?
        x = x.view(x.size(0), self.n_channels, self.img_size[0], self.img_size[1])
        #print("Size:", x.shape)
        return x

class VAE(nn.Module):
    """ Variational Auto-Encoder class. """

    def __init__(self, z_dim, n_channels=3, img_size=[64,64]):
        """
        img_size: [n_channels, height, width] or [height, width]. If given as length 3, the n_channels variable will be ignored.
        n_channels: Number of channels in the input image.
        """
        super().__init__()
        
        self.latent_dim = z_dim
        if len(img_size) == 3:
            self.n_channels = img_size[0]
            self.img_size = img_size[1:]
        else: 
            self.n_channels = n_channels
            self.img_size = img_size

        self.encoder = ResNet18Enc(z_dim=self.latent_dim, n_channels=self.n_channels)
        self.decoder = ResNet18Dec(z_dim=self.latent_dim, img_size=self.img_size, n_channels=self.n_channels)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, mean, logvar
    
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def get_latents(self, x):
        """ Encode data x into latent representations z. """
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return z