# -*- coding:utf-8 -*-
# Created Time: 2018/05/11 10:21:32
# Author: Taihong Xiao <xiaotaihong@126.com>

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(200, 512, 4, 2, 0),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            nn.ConvTranspose3d(512, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x's size: batch_size * hidden_size
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv3d(1, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv3d(128, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2)
        )
        self.layer_4 = nn.Sequential(
            nn.Conv3d(256, 512, 4, 2, 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2)
        )
        self.layer_5 = nn.Sequential(
            nn.Conv3d(512, 1, 4, 2, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x's size: batch_size * 1 * 64 * 64 * 64
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        return x.view(-1, x.size(1))

    def forward_eval(self, x):
        """Forward method for classification, specifically"""
        batch_size = x.size(0)
        x = self.layer_1(x)
        output_2 = self.layer_2(x)
        output_3 = self.layer_3(output_2)
        output_4 = self.layer_4(output_3)

        output_2 = F.max_pool3d(output_2, 8)  # batch x 128 x 2 x 2 x 2
        output_3 = F.max_pool3d(output_3, 4)  # batch x 256 x 2 x 2 x 2
        output_4 = F.max_pool3d(output_4, 2)  # batch x 256 x 2 x 2 x 2
        return torch.cat((output_2.view(batch_size, -1), output_3.view(batch_size, -1), output_4.view(batch_size, -1)), dim=1)  # batch_size x 7168


if __name__ == "__main__":
    G = Generator().cuda(0)
    D = Discriminator().cuda(0)
    G = torch.nn.DataParallel(G, device_ids=[0,1])
    D = torch.nn.DataParallel(D, device_ids=[0,1])

    # z = Variable(torch.rand(16,512,4,4,4))
    # m = nn.ConvTranspose3d(512, 256, 4, 2, 1)
    z = Variable(torch.rand(16, 200, 1,1,1)).cuda(1)
    X = G(z)
    m = nn.Conv3d(1, 64, 4, 2, 1)
    D_X = D(X)
    print(X.shape, D_X.shape)
