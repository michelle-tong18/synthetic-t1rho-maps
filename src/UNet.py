from collections import OrderedDict

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._firstblock(in_channels, features, name="enc1")
        self.encoder2 = UNet._fwdblock(features,features * 2, name = "enc2")
        self.encoder3 = UNet._fwdblock(features * 2,features * 4, name = "enc3")
        self.encoder4 = UNet._fwdblock(features * 4,features * 8, name = "enc4")
        self.encoder5 = UNet._fwdblock(features * 8,features * 8, name = "enc5")
        self.encoder6 = UNet._fwdblock(features * 8,features * 8, name = "enc6")
        self.encoder7 = UNet._fwdblock(features * 8,features * 8, name = "enc7")
        self.encoder8 = UNet._fwdblock_end(features * 8,features * 8, name = "enc8")

        self.decoder8 = UNet._revblock(features * 8,features * 8, name = "dec8")
        self.decoder7 = UNet._revblock((features * 8) * 2,features * 8, name = "dec7")
        self.decoder6 = UNet._revblock((features * 8) * 2,features * 8, name = "dec6")
        self.decoder5 = UNet._revblock((features * 8) * 2,features * 8, name = "dec5")
        self.decoder4 = UNet._revblock((features * 8) * 2,features * 4, name = "dec4")
        self.decoder3 = UNet._revblock((features * 4) * 2,features * 2, name = "dec3")
        self.decoder2 = UNet._revblock((features * 2) * 2,features, name = "dec2")
        self.decoder1 = UNet._lastblock((features) * 2, out_channels, name = "dec1")

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        enc7 = self.encoder7(enc6)
        enc8 = self.encoder8(enc7)
        
        map_dec8 = self.decoder8(enc8)
        map_dec8 = torch.cat((map_dec8,enc7), dim = 1)
        map_dec7 = self.decoder7(map_dec8)
        map_dec7 = torch.cat((map_dec7,enc6), dim = 1)
        map_dec6 = self.decoder6(map_dec7)
        map_dec6 = torch.cat((map_dec6,enc5), dim = 1)
        map_dec5 = self.decoder5(map_dec6)
        map_dec5 = torch.cat((map_dec5,enc4), dim = 1)
        map_dec4 = self.decoder4(map_dec5)
        map_dec4 = torch.cat((map_dec4,enc3), dim = 1)
        map_dec3 = self.decoder3(map_dec4)
        map_dec3 = torch.cat((map_dec3,enc2), dim = 1)
        map_dec2 = self.decoder2(map_dec3)
        map_dec2 = torch.cat((map_dec2,enc1), dim = 1)
        map_dec1 = self.decoder1(map_dec2)
        
        return map_dec1

    @staticmethod
    def _firstblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=4,
                            stride = 2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )
    
    
    @staticmethod
    def _fwdblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "relu1", nn.ReLU(inplace=False)),
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=4,
                            stride = 2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )
    
    @staticmethod
    def _fwdblock_end(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "relu1", nn.ReLU(inplace=False)),
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=4,
                            stride = 2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                ]
            )
        )
    
    @staticmethod
    def _revblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "relu1", nn.ReLU(inplace=False)),
                    (
                        name + "conv1",
                        nn.ConvTranspose2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=4,
                            stride = 2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )
    
    @staticmethod
    def _lastblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "relu1", nn.ReLU(inplace=False)),
                    (
                        name + "conv1",
                        nn.ConvTranspose2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=4,
                            stride = 2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu2", nn.ReLU(inplace=False)),
                ]
            )
        )