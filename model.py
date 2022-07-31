import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init_normal
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator_S2F(nn.Module):
    def __init__(self,init_weights=False):
        super(Generator_S2F, self).__init__()

        # Initial convolution block
        self.conv1_b=nn.Sequential(nn.ReflectionPad2d(3),
                    nn.Conv2d(3, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True))
        self.downconv2_b=nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(128),
                                    nn.ReLU(inplace=True))
        self.downconv3_b=nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.conv4_b=nn.Sequential(ResidualBlock(256))
        self.conv5_b=nn.Sequential(ResidualBlock(256))
        self.conv6_b=nn.Sequential(ResidualBlock(256))
        self.conv7_b=nn.Sequential(ResidualBlock(256))
        self.conv8_b=nn.Sequential(ResidualBlock(256))
        self.conv9_b=nn.Sequential(ResidualBlock(256))
        self.conv10_b=nn.Sequential(ResidualBlock(256))
        self.conv11_b=nn.Sequential(ResidualBlock(256))
        self.conv12_b=nn.Sequential(ResidualBlock(256))
        self.upconv13_b=nn.Sequential(nn.ConvTranspose2d(256,128,3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True))
        self.upconv14_b=nn.Sequential(nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU(inplace=True))
        self.conv15_b=nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(64, 3, 7))
        
        if init_weights:
            self.apply(weights_init_normal)
    
    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_S2F(init_weights=True)
        return model


    def forward(self,xin):
        x=self.conv1_b(xin)
        x=self.downconv2_b(x)
        x=self.downconv3_b(x)
        x=self.conv4_b(x)
        x=self.conv5_b(x)
        x=self.conv6_b(x)
        x=self.conv7_b(x)
        x=self.conv8_b(x)
        x=self.conv9_b(x)
        x=self.conv10_b(x)
        x=self.conv11_b(x)
        x=self.conv12_b(x)
        x=self.upconv13_b(x)
        x=self.upconv14_b(x)
        x=self.conv15_b(x)
        xout=x+xin
        return xout.tanh()

class Generator_F2S(nn.Module):
    def __init__(self,init_weights=False):
        super(Generator_F2S, self).__init__()

        # Initial convolution block
        self.conv1_b=nn.Sequential(nn.ReflectionPad2d(3),
                    nn.Conv2d(4, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True))
        self.downconv2_b=nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(128),
                                    nn.ReLU(inplace=True))
        self.downconv3_b=nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.conv4_b=nn.Sequential(ResidualBlock(256))
        self.conv5_b=nn.Sequential(ResidualBlock(256))
        self.conv6_b=nn.Sequential(ResidualBlock(256))
        self.conv7_b=nn.Sequential(ResidualBlock(256))
        self.conv8_b=nn.Sequential(ResidualBlock(256))
        self.conv9_b=nn.Sequential(ResidualBlock(256))
        self.conv10_b=nn.Sequential(ResidualBlock(256))
        self.conv11_b=nn.Sequential(ResidualBlock(256))
        self.conv12_b=nn.Sequential(ResidualBlock(256))
        self.upconv13_b=nn.Sequential(nn.ConvTranspose2d(256,128,3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True))
        self.upconv14_b=nn.Sequential(nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU(inplace=True))
        self.conv15_b=nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(64, 3, 7))
        
        if init_weights:
            self.apply(weights_init_normal)
    
    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_F2S(init_weights=True)
        return model

    def forward(self,xin,mask):
        x=torch.cat((xin,mask),1)
        x=self.conv1_b(x)
        x=self.downconv2_b(x)
        x=self.downconv3_b(x)
        x=self.conv4_b(x)
        x=self.conv5_b(x)
        x=self.conv6_b(x)
        x=self.conv7_b(x)
        x=self.conv8_b(x)
        x=self.conv9_b(x)
        x=self.conv10_b(x)
        x=self.conv11_b(x)
        x=self.conv12_b(x)
        x=self.upconv13_b(x)
        x=self.upconv14_b(x)
        x=self.conv15_b(x)
        xout=x+xin
        return xout.tanh()


class Generator_S2F_noDown(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_S2F_noDown, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(3, 64, 7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(64),
                                         nn.ReLU(inplace=True))
        # self.downconv3_b = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
        #                                  nn.InstanceNorm2d(256),
        #                                  nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(64))
        self.conv5_b = nn.Sequential(ResidualBlock(64))
        self.conv6_b = nn.Sequential(ResidualBlock(64))
        self.conv7_b = nn.Sequential(ResidualBlock(64))
        self.conv8_b = nn.Sequential(ResidualBlock(64))
        self.conv9_b = nn.Sequential(ResidualBlock(64))
        self.conv10_b = nn.Sequential(ResidualBlock(64))
        self.conv11_b = nn.Sequential(ResidualBlock(64))
        self.conv12_b = nn.Sequential(ResidualBlock(64))
        # self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
        #                                 nn.InstanceNorm2d(128),
        #                                 nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_S2F_noDown(init_weights=True)
        return model

    def forward(self, xin):
        x = self.conv1_b(xin)
        x = self.downconv2_b(x)
        # x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv6_b(x)
        x = self.conv7_b(x)
        x = self.conv8_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        # x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()


class Generator_F2S_noDown(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_F2S_noDown, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(4, 64, 7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        # self.downconv3_b = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
        #                                  nn.InstanceNorm2d(256),
        #                                  nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(64))
        self.conv5_b = nn.Sequential(ResidualBlock(64))
        self.conv6_b = nn.Sequential(ResidualBlock(64))
        self.conv7_b = nn.Sequential(ResidualBlock(64))
        self.conv8_b = nn.Sequential(ResidualBlock(64))
        self.conv9_b = nn.Sequential(ResidualBlock(64))
        self.conv10_b = nn.Sequential(ResidualBlock(64))
        self.conv11_b = nn.Sequential(ResidualBlock(64))
        self.conv12_b = nn.Sequential(ResidualBlock(64))
        # self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
        #                                 nn.InstanceNorm2d(128),
        #                                 nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_F2S_noDown(init_weights=True)
        return model

    def forward(self, xin, mask):
        x = torch.cat((xin, mask), 1)
        x = self.conv1_b(x)
        x = self.downconv2_b(x)
        # x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv6_b(x)
        x = self.conv7_b(x)
        x = self.conv8_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        # x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()


class Generator_S2F_inCo(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_S2F_inCo, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(19, 64, 7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(256),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(256))
        self.conv5_b = nn.Sequential(ResidualBlock(256))
        self.conv6_b = nn.Sequential(ResidualBlock(256))
        self.conv7_b = nn.Sequential(ResidualBlock(256))
        self.conv8_b = nn.Sequential(ResidualBlock(256))
        self.conv9_b = nn.Sequential(ResidualBlock(256))
        self.conv10_b = nn.Sequential(ResidualBlock(256))
        self.conv11_b = nn.Sequential(ResidualBlock(256))
        self.conv12_b = nn.Sequential(ResidualBlock(256))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_S2F_inCo(init_weights=True)
        return model

    def forward(self, nonshadow_feature, xin):
        x = torch.cat((nonshadow_feature, xin), 1)
        x = self.conv1_b(x)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv6_b(x)
        x = self.conv7_b(x)
        x = self.conv8_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()


class Generator_F2S_inCo(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_F2S_inCo, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(20, 64, 7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(256),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(256))
        self.conv5_b = nn.Sequential(ResidualBlock(256))
        self.conv6_b = nn.Sequential(ResidualBlock(256))
        self.conv7_b = nn.Sequential(ResidualBlock(256))
        self.conv8_b = nn.Sequential(ResidualBlock(256))
        self.conv9_b = nn.Sequential(ResidualBlock(256))
        self.conv10_b = nn.Sequential(ResidualBlock(256))
        self.conv11_b = nn.Sequential(ResidualBlock(256))
        self.conv12_b = nn.Sequential(ResidualBlock(256))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_F2S_inCo(init_weights=True)
        return model

    def forward(self, nonshadow_feature, xin, mask):
        x = torch.cat((nonshadow_feature, xin, mask), 1)
        x = self.conv1_b(x)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv6_b(x)
        x = self.conv7_b(x)
        x = self.conv8_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()

class Generator_S2F_128(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_S2F_128, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(3, 64, 7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(128))
        self.conv5_b = nn.Sequential(ResidualBlock(128))
        self.conv6_b = nn.Sequential(ResidualBlock(128))
        self.conv7_b = nn.Sequential(ResidualBlock(128))
        self.conv8_b = nn.Sequential(ResidualBlock(128))
        self.conv9_b = nn.Sequential(ResidualBlock(128))
        self.conv10_b = nn.Sequential(ResidualBlock(128))
        self.conv11_b = nn.Sequential(ResidualBlock(128))
        self.conv12_b = nn.Sequential(ResidualBlock(128))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_S2F_128(init_weights=True)
        return model

    def forward(self, xin):
        x = self.conv1_b(xin)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv6_b(x)
        x = self.conv7_b(x)
        x = self.conv8_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()


class Generator_F2S_128(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_F2S_128, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(4, 64, 7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(128))
        self.conv5_b = nn.Sequential(ResidualBlock(128))
        self.conv6_b = nn.Sequential(ResidualBlock(128))
        self.conv7_b = nn.Sequential(ResidualBlock(128))
        self.conv8_b = nn.Sequential(ResidualBlock(128))
        self.conv9_b = nn.Sequential(ResidualBlock(128))
        self.conv10_b = nn.Sequential(ResidualBlock(128))
        self.conv11_b = nn.Sequential(ResidualBlock(128))
        self.conv12_b = nn.Sequential(ResidualBlock(128))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_F2S_128(init_weights=True)
        return model

    def forward(self, xin, mask):
        x = torch.cat((xin, mask), 1)
        x = self.conv1_b(x)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv6_b(x)
        x = self.conv7_b(x)
        x = self.conv8_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()

class Generator_S2F_64(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_S2F_64, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(3, 64, 7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(64))
        self.conv5_b = nn.Sequential(ResidualBlock(64))
        self.conv6_b = nn.Sequential(ResidualBlock(64))
        self.conv7_b = nn.Sequential(ResidualBlock(64))
        self.conv8_b = nn.Sequential(ResidualBlock(64))
        self.conv9_b = nn.Sequential(ResidualBlock(64))
        self.conv10_b = nn.Sequential(ResidualBlock(64))
        self.conv11_b = nn.Sequential(ResidualBlock(64))
        self.conv12_b = nn.Sequential(ResidualBlock(64))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_S2F_64(init_weights=True)
        return model

    def forward(self, xin):
        x = self.conv1_b(xin)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv6_b(x)
        x = self.conv7_b(x)
        x = self.conv8_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()


class Generator_F2S_64(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_F2S_64, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(4, 64, 7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(64))
        self.conv5_b = nn.Sequential(ResidualBlock(64))
        self.conv6_b = nn.Sequential(ResidualBlock(64))
        self.conv7_b = nn.Sequential(ResidualBlock(64))
        self.conv8_b = nn.Sequential(ResidualBlock(64))
        self.conv9_b = nn.Sequential(ResidualBlock(64))
        self.conv10_b = nn.Sequential(ResidualBlock(64))
        self.conv11_b = nn.Sequential(ResidualBlock(64))
        self.conv12_b = nn.Sequential(ResidualBlock(64))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_F2S_64(init_weights=True)
        return model

    def forward(self, xin, mask):
        x = torch.cat((xin, mask), 1)
        x = self.conv1_b(x)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv6_b(x)
        x = self.conv7_b(x)
        x = self.conv8_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()

class Generator_Encoder(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_Encoder, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(3, 64, 7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(128))
        self.conv5_b = nn.Sequential(ResidualBlock(128))
        self.conv6_b = nn.Sequential(ResidualBlock(128))
        self.conv7_b = nn.Sequential(ResidualBlock(128))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_Encoder(init_weights=True)
        return model

    def forward(self, xin):
        x = self.conv1_b(xin)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv6_b(x)
        x = self.conv7_b(x)
        xout = x
        return xout


class Generator_decoder(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_decoder, self).__init__()

        self.conv1_b = nn.Sequential(nn.Conv2d(256, 256, 1),
                                     nn.InstanceNorm2d(256),
                                     nn.ReLU(inplace=True))
        # self.conv2_b = nn.Sequential(nn.Conv2d(256, 256, 1),
        #                              nn.InstanceNorm2d(256),
        #                              nn.ReLU(inplace=True))
        self.conv9_b = nn.Sequential(ResidualBlock(256))
        self.conv10_b = nn.Sequential(ResidualBlock(256))
        self.conv11_b = nn.Sequential(ResidualBlock(256))
        self.conv12_b = nn.Sequential(ResidualBlock(256))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_decoder(init_weights=True)
        return model

    def forward(self, xin, detail_feature):
        x = torch.cat((xin, detail_feature), 1)
        x = self.conv1_b(x)
        # x = self.conv2_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x
        return xout.tanh()


class Generator_Encoder_SC(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_Encoder_SC, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(3, 16, 7),
                                     nn.InstanceNorm2d(16),
                                     nn.ReLU(inplace=True))
        self.conv3_b = nn.Sequential(ResidualBlock(16))
        self.conv4_b = nn.Sequential(ResidualBlock(16))
        # self.conv5_b = nn.Sequential(ResidualBlock(16))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_Encoder_SC(init_weights=True)
        return model

    def forward(self, xin):
        x = self.conv1_b(xin)
        x = self.conv3_b(x)
        x = self.conv4_b(x)
        # x = self.conv5_b(x)
        xout = x
        return xout

class Generator_decoder_SC(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_decoder_SC, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(19, 128, 7),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(128))
        self.conv5_b = nn.Sequential(ResidualBlock(128))
        self.conv9_b = nn.Sequential(ResidualBlock(128))
        self.conv10_b = nn.Sequential(ResidualBlock(128))
        self.conv11_b = nn.Sequential(ResidualBlock(128))
        self.conv12_b = nn.Sequential(ResidualBlock(128))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_decoder_SC(init_weights=True)
        return model

    def forward(self, shadow_feature, xin):
        x = torch.cat((shadow_feature, xin), 1)
        x = self.conv1_b(x)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        # x = self.conv6_b(x)
        # x = self.conv7_b(x)
        # x = self.conv8_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()

class Generator_decoder_SC_NSen(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_decoder_SC_NSen, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(3, 16, 7),
                                     nn.InstanceNorm2d(16),
                                     nn.ReLU(inplace=True))
        self.conv1_b1 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(32, 128, 7),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(128))
        self.conv5_b = nn.Sequential(ResidualBlock(128))
        self.conv9_b = nn.Sequential(ResidualBlock(128))
        self.conv10_b = nn.Sequential(ResidualBlock(128))
        self.conv11_b = nn.Sequential(ResidualBlock(128))
        self.conv12_b = nn.Sequential(ResidualBlock(128))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_decoder_SC_NSen(init_weights=True)
        return model

    def forward(self, shadow_feature, xin):
        # x = torch.cat((shadow_feature, xin), 1)
        x = self.conv1_b(xin)
        x = torch.cat((shadow_feature, x), 1)
        x = self.conv1_b1(x)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()

class Generator_decoder_SC_noDown(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_decoder_SC_noDown, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(19, 64, 7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(64))
        self.conv5_b = nn.Sequential(ResidualBlock(64))
        self.conv9_b = nn.Sequential(ResidualBlock(64))
        self.conv10_b = nn.Sequential(ResidualBlock(64))
        self.conv11_b = nn.Sequential(ResidualBlock(64))
        self.conv12_b = nn.Sequential(ResidualBlock(64))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_decoder_SC_noDown(init_weights=True)
        return model

    def forward(self, shadow_feature, xin):
        x = torch.cat((shadow_feature, xin), 1)
        x = self.conv1_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()

class Generator_decoder_SC_contrast(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_decoder_SC_contrast, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(20, 128, 7),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(128))
        self.conv5_b = nn.Sequential(ResidualBlock(128))
        # self.conv6_b = nn.Sequential(ResidualBlock(128))
        # self.conv7_b = nn.Sequential(ResidualBlock(128))
        # self.conv8_b = nn.Sequential(ResidualBlock(128))
        self.conv9_b = nn.Sequential(ResidualBlock(128))
        self.conv10_b = nn.Sequential(ResidualBlock(128))
        self.conv11_b = nn.Sequential(ResidualBlock(128))
        self.conv12_b = nn.Sequential(ResidualBlock(128))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_decoder_SC_contrast(init_weights=True)
        return model

    def forward(self, shadow_feature, xin, mask):
        x = torch.cat((shadow_feature, xin, mask), 1)
        x = self.conv1_b(x)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        # x = self.conv6_b(x)
        # x = self.conv7_b(x)
        # x = self.conv8_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(3, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self,x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1).squeeze() #global avg pool

def define_models():
    # Networks
    netG_A2B = Generator_decoder_SC()
    # netG_A2B = Generator_decoder_SC_noDown()
    # netG_A2B = Generator_decoder_SC_NSen()
    netEn_sf = Generator_Encoder_SC()

    # netD_B = Discriminator()
    netD_S = Discriminator()
    netD_NS = Discriminator()
    netG_1 = Generator_S2F()  # shadow to shadow_free
    netG_2 = Generator_F2S()  # shadow to shadow_free

    netG_A2B.cuda()
    netEn_sf.cuda()
    netD_S.cuda()
    netD_NS.cuda()
    netG_1.cuda()
    netG_2.cuda()

    netG_A2B.apply(weights_init_normal)
    netEn_sf.apply(weights_init_normal)
    netD_S.apply(weights_init_normal)
    netD_NS.apply(weights_init_normal)
    netG_1.apply(weights_init_normal)
    netG_2.apply(weights_init_normal)

    return netG_A2B, netEn_sf, netD_S, netD_NS, netG_1, netG_2