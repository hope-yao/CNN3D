#!/usr/bin/env python


from __future__ import division, print_function

import sys

sys.path.append("../lib")

import logging
import theano
import theano.tensor as T
from theano import tensor
import numpy as np

from blocks.bricks.base import application, lazy
from blocks.bricks.recurrent import BaseRecurrent, recurrent, LSTM
from blocks.bricks import Random, Initializable, MLP, Linear, Rectifier
from blocks.bricks.parallel import Parallel, Fork
from blocks.bricks import Tanh, Identity, Softmax, Logistic
from fuel.datasets.hdf5 import H5PYDataset

from blocks.initialization import Constant, IsotropicGaussian, Orthogonal, Uniform
# from bricks3D.cnn3d_bricks import Convolutional3, MaxPooling3, ConvolutionalSequence3, Flattener3
from blocks.bricks.conv import Convolutional, MaxPooling

class conv_RAM(BaseRecurrent, Initializable, Random):
    def __init__(self, image_size, channels, attention, n_iter, **kwargs):
        super(conv_RAM, self).__init__(**kwargs)

        self.n_iter = n_iter
        self.channels = channels
        self.read_N = attention
        self.image_ndim = len(image_size)
        if self.image_ndim == 2:
            self.img_height, self.img_width = image_size
        elif self.image_ndim == 3:
            self.img_height, self.img_width, self.img_depth = image_size
        self.dim_h = 7**3

        l = tensor.matrix('l')  # for a batch
        n_class = 10
        dim_h = self.dim_h
        dim_data = 2
        inits = {
            # 'weights_init': IsotropicGaussian(0.01),
            # 'biases_init': Constant(0.),
            'weights_init': Orthogonal(),
            'biases_init': IsotropicGaussian(),
        }

        # glimpse network
        self.conv_g0 = Convolutional(filter_size=(5,5),num_filters=16, num_channels=1 ,step=(1,1),border_mode='valid',name='conv_g0')
        self.conv_g1 = Convolutional(filter_size=(5,5),num_filters=16, num_channels=1 ,step=(1,1),border_mode='valid',name='conv_g1')
        self.conv_g2 = Convolutional(filter_size=(5,5),num_filters=16, num_channels=1 ,step=(1,1),border_mode='valid',name='conv_g2')
        self.pool_g0 = MaxPooling(pooling_size=(2,2), name='pool_g0')
        self.pool_g1 = MaxPooling(pooling_size=(2,2), name='pool_g1')
        self.pool_g2 = MaxPooling(pooling_size=(2,2), name='pool_g2')
        self.rect_g0 = Rectifier()
        self.rect_g1 = Rectifier()
        self.rect_g2 = Rectifier()
        # self.conv_sequence_g = ConvolutionalSequence3(self.layers, num_channels, image_size=image_shape)
        # core network
        self.conv_h0 = Convolutional(filter_size=(5,5),num_filters=24, num_channels=1 ,step=(1,1),border_mode='valid',name='conv_h0')
        self.conv_h1 = Convolutional(filter_size=(5,5),num_filters=24, num_channels=1 ,step=(1,1),border_mode='valid',name='conv_h1')
        self.conv_h2 = Convolutional(filter_size=(5,5),num_filters=24, num_channels=1 ,step=(1,1),border_mode='valid',name='conv_h2')
        self.pool_h0 = MaxPooling(pooling_size=(2,2), name='pool_h0')
        self.pool_h1 = MaxPooling(pooling_size=(2,2), name='pool_h1')
        self.pool_h2 = MaxPooling(pooling_size=(2,2), name='pool_h2')
        self.rect_h0 = Rectifier()
        self.rect_h1 = Rectifier()
        self.rect_h2 = Rectifier()

        # location network
        self.linear_l = MLP(activations=[Logistic()], dims=[dim_h, dim_data], name="location network", **inits)

        # classification network
        self.linear_a = MLP(activations=[Softmax()], dims=[dim_h, n_class], name="classification network", **inits)

        self.children = [self.conv_g0, self.conv_g1, self.conv_g2, self.pool_g0, self.pool_g1, self.pool_g2, self.rect_g0, self.rect_g1, self.rect_g2,
                         self.conv_h0, self.conv_h1, self.conv_h2, self.pool_h0, self.pool_h1, self.pool_h2,self.rect_h0, self.rect_h1, self.rect_h2,
                         self.linear_l, self.linear_a]


    @property
    def output_dim(self):
        return 10

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_g0._push_allocation_config()
        self.conv_g1._push_allocation_config()
        self.conv_g2._push_allocation_config()
        self.pool_g0._push_allocation_config()
        self.pool_g1._push_allocation_config()
        self.pool_g2._push_allocation_config()
        self.rect_g0._push_allocation_config()
        self.rect_g1._push_allocation_config()
        self.rect_g2._push_allocation_config()

        self.conv_h0._push_allocation_config()
        self.conv_h1._push_allocation_config()
        self.conv_h2._push_allocation_config()
        self.pool_h0._push_allocation_config()
        self.pool_h1._push_allocation_config()
        self.pool_h2._push_allocation_config()
        self.rect_h0._push_allocation_config()
        self.rect_h1._push_allocation_config()
        self.rect_h2._push_allocation_config()

        self.linear_l._push_allocation_config()
        self.linear_a._push_allocation_config()

    def get_dim(self, name):
        if name == 'prob':
            return 10 # for mnist_lenet
        elif name == 'h':
            return self.dim_h
        elif name == 'l':
            return self.image_ndim
        else:
            super(conv_RAM, self).get_dim(name)

    # ------------------------------------------------------------------------
    @recurrent(sequences=['dummy'], contexts=['x'],
               states=['l', 'h'],
               outputs=['l', 'prob'])  # h seems not necessary
    def apply(self, x, dummy, l=None, h=None):
        if self.image_ndim == 2:
            from theano.tensor.signal.pool import pool_2d
            from attention import ZoomableAttentionWindow

            scale = 1
            zoomer_orig = ZoomableAttentionWindow(self.channels, self.img_height, self.img_width, self.read_N, scale)
            rho_orig = zoomer_orig.read_large(x, l[:,1], l[:,0]) # glimpse sensor in 2D, output matrix

            scale = 2
            zoomer_larger = ZoomableAttentionWindow(self.channels, self.img_height, self.img_width, self.read_N, scale)
            rho_larger = zoomer_larger.read_large(x, l[:, 1], l[:, 0])  # glimpse sensor in 2D

            scale = 4
            zoomer_largest = ZoomableAttentionWindow(self.channels, self.img_height, self.img_width, self.read_N, scale)
            rho_largest = zoomer_largest.read_large(x, l[:, 1], l[:, 0])  # glimpse sensor in 2D

        elif self.image_ndim == 3:
            from attention import ZoomableAttentionWindow3d
            zoomer = ZoomableAttentionWindow3d(self.channels, self.img_height, self.img_width, self.img_depth, self.read_N)
            rho = zoomer.read_large(x, l[:,0], l[:,1], l[:,2]) # glimpse sensor in 3D

        # glimpse network
        tt = self.conv_g0.apply(rho_orig)
        g0 = self.rect_g0.apply(self.pool_g0.apply(tt))
        g1 = self.rect_g1.apply(self.pool_g1.apply(self.conv_g1.apply(rho_larger)))
        g2 = self.rect_g2.apply(self.pool_g2.apply(self.conv_g2.apply(rho_largest)))

        # core network
        h0 = self.rect_h0.apply(self.pool_h0.apply(self.conv_h0.apply(g0)))
        h1 = self.rect_h1.apply(self.pool_h1.apply(self.conv_h1.apply(g1)))
        h2 = self.rect_h2.apply(self.pool_h2.apply(self.conv_h2.apply(g2)))
        # h = T.concatenate([h0, h1, h2],axis = 0) # might be wrong

        h_flatten = T.concatenate([h0.flatten(),h1.flatten(),h2.flatten()])
        prob = self.linear_a.apply(h_flatten)
        l = self.linear_l.apply(h_flatten)

        return l, prob

    # ------------------------------------------------------------------------

    @application(inputs=['features'], outputs=['l', 'prob'])
    def classify(self, features):
        batch_size = features.shape[0]
        # No particular use apart from control n_steps in RNN
        u = self.theano_rng.normal(
            size=(self.n_iter, batch_size, 1),
            avg=0., std=1.)
        bs = 16
        img = 28
        # if self.image_ndim == 2:
        #     center_y_ = T.ones((bs,)) * img/2
        #     center_x_ = T.ones((bs,)) * img/2
        #     init_l = [center_x_, center_y_]
        # else:
        #     center_x_ = T.vector()
        #     center_y_ = T.vector()
        #     center_z_ = T.vector()
        #     init_l = [center_x_, center_y_, center_z_]
        # init_l = tensor.matrix('l')  # for a batch
        l, prob = self.apply(x=features, dummy=u)

        return l, prob

if __name__ == "__main__":

    # ----------------------------------------------------------------------

    ram = conv_RAM(image_size=(28,28), channels=1, attention=5, n_iter=4)
    ram.push_initialization_config()
    # ram.initialize()
    # ------------------------------------------------------------------------
    x = tensor.ftensor4('features')  # keyword from fuel
    y = tensor.matrix('targets')  # keyword from fuel
    l, prob = ram.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.

    f = theano.function([x], [l, prob])
    # test single forward pass
    mnist_train = H5PYDataset('./data/mnist.hdf5', which_sets=('train',))
    handle = mnist_train.open()
    train_data = mnist_train.get_data(handle, slice(0, 16))
    xx = train_data[0]
    print(xx.shape)
    l, prob = f(xx)
    print(l)
    print(prob)