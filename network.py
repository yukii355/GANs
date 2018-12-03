
import torch
import torch.nn as nn
import torch.nn.functional as F


class generator(nn.Module):


    def __init__(self, ngpu, ngf, nz, nc):
        super(generator, self).__init__()

        self.ngpu = ngpu
        self.ngf = ngf # number of filters in generator's first layer

        self.train = nn.Sequential(
            nn.ConvTranspose2d(nz, self.ngf * 8, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),
            # state size: (ngf * 8) * 4 * 4

            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),
            # state size: (ngf * 4) * 8 * 8

            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),
            # state size: (ngf * 2) * 16 * 16

            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),
            # state size: ngf * 32 * 32

            nn.ConvTranspose2d(self.ngf, nc, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size: out_size * 64 * 64
        )


    def forward(self, input):
        output = self.train(input)
        return output


class discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(discriminator, self).__init__()
        self.ngpu = ngpu


        self.train = nn.Sequential(
            # input size is in_size * 64 * 64
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf * 32 * 32

            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 2 ) * 16 * 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 4 ) * 8 * 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 8 ) * 4 * 4

            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size: 1 * 1 * 1
        )

    def forward(self, input):
        # output = self.train(input)
        output = nn.parallel.data_parallel(self.train, input, range(self.ngpu))

        return output





