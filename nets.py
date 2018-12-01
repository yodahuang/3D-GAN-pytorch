# -*- coding:utf-8 -*-
# Created Time: 2018/05/11 10:21:32
# Author: Taihong Xiao <xiaotaihong@126.com>

import torch
import torch.nn as nn
from torch.autograd import Variable


# The following attention module is stolen from https://github.com/heykeetae/Self-Attention-GAN
# With modifications that change 2d conv to 3d conv
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X D X H X W)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Depth*Width*Height)
        """
        m_batchsize, C, depth, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, depth*width*height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, depth*width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        del proj_query, proj_key  # Save some CUDA memory
        attention = self.softmax(energy)  # BX (N) X (N)
        del energy  # save some CUDA memory
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, depth*width*height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, depth, width, height)

        out = self.gamma*out + x
        return out, attention


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.l1 = nn.Sequential(
            nn.ConvTranspose3d(200, 512, 4, 2, 0),
            nn.BatchNorm3d(512),
            nn.ReLU(),
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.l4 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.last = nn.Sequential(
            nn.ConvTranspose3d(64, 1, 4, 2, 1),
            nn.Sigmoid()
        )
        self.attn1 = Self_Attn(128, 'relu')
        self.attn2 = Self_Attn(64,  'relu')

    def forward(self, x):
        # x's size: batch_size * hidden_size
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.last(out)
        return out, p1, p2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv3d(1, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2)
        )
        self.l2 = nn.Sequential(
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2)
        )
        self.l3 = nn.Sequential(
            nn.Conv3d(128, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2)
        )
        self.l4 = nn.Sequential(
            nn.Conv3d(256, 512, 4, 2, 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2)
        )
        self.last = nn.Sequential(
            nn.Conv3d(512, 1, 4, 2, 0),
            nn.Sigmoid()
        )
        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        # x's size: batch_size * 1 * 64 * 64 * 64
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.last(out)
        return out.view(-1, out.size(1)), p1, p2


if __name__ == "__main__":
    G = Generator().cuda(0)
    D = Discriminator().cuda(0)
    G = torch.nn.DataParallel(G, device_ids=[0, 1])
    D = torch.nn.DataParallel(D, device_ids=[0, 1])

    # z = Variable(torch.rand(16,512,4,4,4))
    # m = nn.ConvTranspose3d(512, 256, 4, 2, 1)
    z = Variable(torch.rand(16, 200, 1, 1, 1)).cuda(1)
    X = G(z)
    m = nn.Conv3d(1, 64, 4, 2, 1)
    D_X = D(X)
    print(X.shape, D_X.shape)
