import torch.nn as nn
import torch
import torch.nn.functional as F
import dsntnn
import torchvision


class Encoder(nn.Module):
    """
    Encoding part of the UNet
    """
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        """
        :param chs:     Channel size for the encoding convolutional layers
        """
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        """
        :param chs:     Channel size for the decoding convolutional layers
        """
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    @staticmethod
    def crop(enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        :param in_ch:       Input channels of first convolutional layer
        :param out_ch:      Output channel of first, input and output of second convolutional layer
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 5, padding=2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UNet(nn.Module):
    def __init__(self, enc_chs=(1, 16, 32, 64, 128), dec_chs=(128, 64, 32, 16), num_out_ch=1,
                 retain_dim=True):
        """
        :param enc_chs:         Channel sizes of encoding convolutional layers
        :param dec_chs:         Channel sizes of decoding convolutional layers
        :param num_out_ch:      Number of output channels (1 per keypoint)
        :param retain_dim:      Interpolates solution to original image size
        """
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_out_ch, 1)
        self.retain_dim = retain_dim

    def forward(self, x):
        b, c, h, w = x.shape
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, (h, w))
        return out


class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_out):
        """
        :param n_out:       Amount of keypoints to predict
        """
        super().__init__()
        self.n_out = n_out
        self.model = UNet(num_out_ch=n_out)

    def forward(self, images):
        # Predict heatmaps
        unnormalized_heatmaps = self.model(images)
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # Create coords in range [-1, 1] using dsntnn
        coords = dsntnn.dsnt(heatmaps)
        return coords, heatmaps
