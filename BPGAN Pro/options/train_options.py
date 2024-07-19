# -*- coding: utf-8 -*-
"""
Created on Mon Sep 9, 2019
@author: Liming Xu and He Zhang
"""

from .base_options import BaseOptions

# define the options for training.
class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of input images')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavior during training and test.
        parser.add_argument('--eval', action='store_false', help='use eval mode during training time.')
        parser.add_argument('--epoch_count',type=int, default=1,help='指定了衰减开始的迭代轮数。在这之前，学习率将保持不变。')
        parser.add_argument('--niter',type=int, default=10,help='总共的迭代轮数')
        parser.add_argument('--niter_decay',type=int, default=9,help='学习率衰减的总轮数')
        parser.add_argument('--pool_size',type=int, default=2)
        parser.add_argument('--lr',type=float, default=1e-6)
        parser.add_argument('--beta1',type=float, default=0.9)
        parser.add_argument('--lr_policy',type=str, default="linear")
        parser.add_argument('--gan_mode',type=str, default="vanilla")
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.isTrain = True
        return parser
