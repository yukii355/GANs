#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function


import torch
import os 
import random
import visdom
import numpy as np
import argparse # argparseを使うときのmodule
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.optim as optim
import torch.utils.data


from dataloader import image_dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
    parser.add_argument('--dataroot', default="/home/moriyama/real_images/", required=True, help='path to dataset')
    parser.add_argument('--out', help='Directory to output the result')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--batchsize', default=64, help='input batch size')
    parser.add_argument('--imagesize', default=64, help='the height / width of the input image to network')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--nz', default=100, help='Number of hidden units(z)')
    parser.add_argument('--epochs', default=100, help='number of epochs to train for')
    parser.add_argument('--ngpu', default=1, help='number of GPUs to use')
    parser.add_argument('--cuda', help='enables cuda')


    opt=parser.parse_args()
    print(opt)


    if opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imagesize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                               ]))


    else:
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.imagesize),
                                       transforms.CenterCrop(opt.imagesize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                   ]))


    '''
    GPUの使用について（例）
    x = torch.randn(10)
    y = torch.randn(10)

    x = x.to('cuda')
    y = y.to('cuda:0') # cudaの後に：数字で対応したGPUを使用

    z = x * y
    z = z.to('cpu') # cpuへ
    '''


    G_net = generator(ngf=opt.ngf, nc=3, nz=100).to(device)
    D_net = discriminator(ndf=opt.ndf, nc=3).to(device)







    ##################

    ### ここから上までで、モデルの構築

    ##################



    # Initialize BCELoss function
    criterion = nn.BCELoss()


    assert dataset

    # dataset = dset.ImageFolder(root = opt.dataroot)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize, shuffle=True)


    # setup optimizer
    optimizerG = torch.optim.Adam(G_net.parameters(), lr=opt.lr)
    optimizerD = torch.optim.Adam(D_net.parameters(), lr=opt.lr)

    # Generate batch of latent vectors
    input_noise = torch.randn(opt.batchsize, opt.nz, 1, 1, device=device)  # batch * channels * height * width

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Lists to keep track of progress
    G_loss_list = []
    D_loss_list = []



    print("Starting training loop...")

    '''
    ＜DCGANのアルゴリズム＞
    1.ミニバッチサイズm個のノイズz1, z2, ..., zmをPg(z)から取り出す(生成する)
    2.ミニバッチサイズm個のサンプルx1, x2, ..., xmをデータ生成分布Pdata(x)から取り出す
    3.1/m sigma((log(D(x))) + (log(1 - D(G(z)))))式の、theta(d)における確率的勾配を上るようにDを更新
    4.上記までをk回くりかえす
    5.ミニバッチサイズm個のノイズz1, z2, ..., zmをPg(z)から取り出す
    6.1/m sigma(log(1 - D(G(z)))))式の、theta(g)における確率的勾配を下るようにGを更新
    7.ここまで全てを、訓練回数分だけ繰り返す
    (8.)Dを十分な回数(k回)更新した上で、Gを1回更新することで、常に鑑別機が新しいGの状態に適用できるように学習を進める
    '''


    # For each epoch
    for epoch in range(opt.epochs):
        # For each batch in the dataloader
        for i, (imgs, _) in enumerate(dataloader, 0):
            #######################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))
            #######################
            ## Train with all-real batch




            # configure training data
            real_img = Variable(imgs).to(device)

            # 勾配の初期化
            D_net.zero_grad()

            # Forward pass through D

            real_label = Variable(torch.ones(opt.batchsize)).to(device)

            fake_label = Variable(torch.zeros(opt.batchsize)).to(device)

            real_out = D_net(real_img)

            # Calculate loss on all-real batch
            d_loss_real = criterion(real_out, real_label)




            ## Train with all-fake batch

            fake_img = G_net(input_noise)
            fake_out = D_net(fake_img)

            d_loss_fake = criterion(fake_img, fake_label)


            # Add the gradients from the all-real and all-fake batches
            d_loss = d_loss_real * d_loss_fake

            # Calculate gradients for D in backward pass
            d_loss.backward()
            # Update D
            optimizerD.step()


            #######################
            # (2) Update G network: maximize log(D(G(z))
            #######################
            G_net.zero_grad()

            fake_img = G_net(input_noise)
            fake_out = D_net(fake_img)


            g_loss = criterion(fake_img, real_img)

            # Calculate gradients for G in backward pass
            g_loss.backward()
            optimizerG.step()


            # Output training stats
            if i % 50 == 0:
                print()

            # Save Losses for plotting later
            G_loss_list.append(g_loss.item())
            D_loss_list.append(d_loss.item())



if __name__ == '__main__':
    main()




