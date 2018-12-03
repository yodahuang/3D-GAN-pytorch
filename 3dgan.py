# -*- coding:utf-8 -*-
# Created Time: 2018/05/11 11:50:23
# Author: Taihong Xiao <xiaotaihong@126.com>

from dataset import config, ShapeNet
from nets import Generator, Discriminator

import os, argparse
import torch
import numpy as np
import scipy.io as sio
from tensorboardX import SummaryWriter
from itertools import chain
from datetime import datetime


class _3DGAN(object):
    def __init__(self, args, config=config):
        self.args = args
        self.attribute = args.attribute
        self.gpu = args.gpu
        self.mode = args.mode
        self.restore = args.restore

        # init dataset and networks
        self.config = config
        if args.batch_size is not None:
            self.config.set_batchsize(args.batch_size)
        self.dataset = ShapeNet(self.attribute)
        self.G = Generator()
        self.D = Discriminator()

        self.adv_criterion = torch.nn.BCELoss()

        self.set_mode_and_gpu()
        self.restore_from_file()

    def set_mode_and_gpu(self):
        if self.mode == 'train':
            self.G.train()
            self.D.train()
            if self.gpu:
                with torch.cuda.device(self.gpu[0]):
                    self.G.cuda()
                    self.D.cuda()
                    self.adv_criterion.cuda()

            if len(self.gpu) > 1:
                self.G = torch.nn.DataParallel(self.G, device_ids=self.gpu)
                self.D = torch.nn.DataParallel(self.D, device_ids=self.gpu)

        elif self.mode == 'test':
            self.G.eval()
            self.D.eval()
            if self.gpu:
                with torch.cuda.device(self.gpu[0]):
                    self.G.cuda()
                    self.D.cuda()

            if len(self.gpu) > 1:
                self.G = torch.nn.DataParallel(self.G, device_ids=self.gpu)
                self.D = torch.nn.DataParallel(self.D, device_ids=self.gpu)

        else:
            raise NotImplementationError()

    def restore_from_file(self):
        if self.restore is not None:
            ckpt_file_G = os.path.join(self.config.model_dir, 'G_iter_{:06d}.pth'.format(self.restore))
            assert os.path.exists(ckpt_file_G)
            self.G.load_state_dict(torch.load(ckpt_file_G))

            if self.mode == 'train':
                ckpt_file_D = os.path.join(self.config.model_dir, 'D_iter_{:06d}.pth'.format(self.restore))
                assert os.path.exists(ckpt_file_D)
                self.D.load_state_dict(torch.load(ckpt_file_D))

            self.start_step = self.restore + 1
        else:
            self.start_step = 1

    def save_log(self):
        scalar_info = {
            'err_D': self.err_D,
            'err_G': self.err_G,
            'G_lr'  : self.G_lr_scheduler.get_lr()[0],
            'D_lr'  : self.D_lr_scheduler.get_lr()[0],
        }
        # for key, value in self.err_G.items():
        #     scalar_info['err_G/' + key] = value
        #
        # for key, value in self.err_D.items():
        #     scalar_info['err_D/' + key] = value

        for tag, value in scalar_info.items():
            self.writer.add_scalar(tag, value, self.step)

    def save_img(self, save_num=5):
        for i in range(save_num):
            mdict = {
                'instance': self.fake_X[i,0].data.cpu().numpy()
            }
            sio.savemat(os.path.join(self.config.img_dir, '{:06d}_{:02d}.mat'.format(self.step, i)), mdict)

    def save_model(self):
        torch.save({key: val.cpu() for key, val in self.G.state_dict().items()}, os.path.join(self.config.model_dir, 'G_iter_{:06d}.pth'.format(self.step)))
        torch.save({key: val.cpu() for key, val in self.D.state_dict().items()}, os.path.join(self.config.model_dir, 'D_iter_{:06d}.pth'.format(self.step)))

    def train(self):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(os.path.join(self.config.log_dir, current_time))
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config.G_lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config.D_lr, betas=(0.5, 0.999))
        self.G_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_G, step_size=self.config.step_size, gamma=self.config.gamma)
        self.D_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_D, step_size=self.config.step_size, gamma=self.config.gamma)

        print('Start training')
        Diters = 5
        clamp_lower = -0.01
        clamp_upper = 0.01

        # start training
        for step in range(self.start_step, 1 + self.config.max_iter):
            self.step = step
            self.G_lr_scheduler.step()
            self.D_lr_scheduler.step()

            # clamp parameters to a cube
            for p in self.D.parameters():
                p.data.clamp_(clamp_lower, clamp_upper)

            self.real_X = next(self.dataset.gen(True))
            self.noise = torch.randn(self.config.nchw[0], 200)
            if len(self.gpu):
                with torch.cuda.device(self.gpu[0]):
                    self.real_X = self.real_X.cuda()
                    self.noise  = self.noise.cuda()

            self.fake_X = self.G(self.noise)

            # update D
            self.D_real = self.D(self.real_X)
            self.D_fake = self.D(self.fake_X.detach())
            self.err_D = torch.mean(self.D_real.data) - torch.mean(self.D_fake.data)

            self.opt_D.zero_grad()
            self.D_real.backward(torch.ones_like(self.D_real))
            self.D_fake.backward(torch.ones_like(self.D_fake) * -1)
            self.opt_D.step()

            if step % Diters == 0:
                # update G
                self.D_fake = self.D(self.fake_X)
                self.err_G = torch.mean(self.D_fake.data)
                self.opt_G.zero_grad()
                self.D_fake.backward(torch.ones_like(self.D_fake))
                self.opt_G.step()

            # print('step: {:06d}, loss_D: {:.6f}, loss_G: {:.6f}'.format(self.step, self.loss_D.data.cpu().numpy(), self.loss_G.data.cpu().numpy()))

            if self.step % 100 == 0:
                self.save_log()
                print('Reached step #{}'.format(self.step))

            if self.step % 1000 == 0:
                self.save_img()
                self.save_model()

        print('Finished training!')
        self.writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attribute', type=str, required=True, help='Specify category for training.')
    parser.add_argument('-g', '--gpu', default=[], nargs='+', type=int, help='Specify GPU ids.')
    parser.add_argument('-r', '--restore', default=None, action='store', type=int, help='Specify checkpoint id to restore.')
    parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('-b', '--batch_size', type=int)
    
    args = parser.parse_args()
    print(args)

    model = _3DGAN(args)
    if args.mode == 'train':
        model.train()

