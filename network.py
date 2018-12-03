
import torch
import torch.nn as nn
import torch.nn.functional as F


class generator(nn.Module):


    def __init__(self, nz, nc, ngf=128):
        super(generator, self).__init__()

        self.ngf = ngf # number of filters in generator's first layer

        self.train = nn.Sequential(
            nn.ConvTranspose2d(nz, self.ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),
            # state size: (ngf * 8) * 4 * 4

            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),
            # state size: (ngf * 4) * 8 * 8

            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),
            # state size: (ngf * 2) * 16 * 16

            nn.ConvTranspose2d(self.ngf * 2, self.ngf , kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),
            # state size: ngf * 32 * 32

            nn.ConvTranspose2d(self.ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size: out_size * 64 * 64
        )


    def forward(self, input):
        output = self.train(input)
        return output

nc = 3


class discriminator(nn.Module):
    def __init__(self, nc, ndf=128):
        super(discriminator, self).__init__()
        # self.in_size = in_size
        self.ndf = ndf

        self.train = nn.Sequential(
            # input size is in_size * 64 * 64
            nn.Conv2d(nc, self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf * 32 * 32

            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 2 ) * 16 * 16

            nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 4 ) * 8 * 8

            nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 8 ) * 4 * 4

            nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size: 1 * 1 * 1
        )

    def forward(self, input):
        output = self.train(input)
        return output





