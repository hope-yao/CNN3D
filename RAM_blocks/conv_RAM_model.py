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
from blocks.bricks.conv import Convolutional, MaxPooling, Flattener, ConvolutionalSequence
from theano.tensor.signal.pool import pool_2d


class conv_RAM(BaseRecurrent, Initializable, Random):
    def __init__(self, image_size, channels, attention, n_iter, n_class, **kwargs):
        super(conv_RAM, self).__init__(**kwargs)

        self.n_iter = n_iter
        self.n_class = n_class
        self.channels = channels
        self.read_N = attention
        self.image_ndim = len(image_size)
        if self.image_ndim == 2:
            self.img_height, self.img_width = image_size
        elif self.image_ndim == 3:
            self.img_height, self.img_width, self.img_depth = image_size

        inits = {
            # 'weights_init': IsotropicGaussian(0.01),
            # 'biases_init': Constant(0.),
            'weights_init': Orthogonal(),
            'biases_init': IsotropicGaussian(),
        }
        conv_inits = {
            'weights_init': Uniform(width=.2),
            'biases_init': Constant(0.)
        }

        self.n_channel = 32
        self.filter_size = 5
        # glimpse network
        self.conv_g0 = Convolutional(filter_size=(self.filter_size,self.filter_size),num_filters=self.n_channel, num_channels=1 ,step=(1,1),border_mode='half',name='conv_g0', **conv_inits)
        # self.conv_g1 = Convolutional(filter_size=(5,5),num_filters=16, num_channels=1 ,step=(1,1),border_mode='half',name='conv_g1', **conv_inits)
        # self.conv_g2 = Convolutional(filter_size=(5,5),num_filters=16, num_channels=1 ,step=(1,1),border_mode='half',name='conv_g2', **conv_inits)
        self.pool_g0 = MaxPooling(pooling_size=(2,2), name='pool_g0')
        # self.pool_g1 = MaxPooling(pooling_size=(2,2), name='pool_g1')
        # self.pool_g2 = MaxPooling(pooling_size=(2,2), name='pool_g2')
        self.rect_g0 = Rectifier()
        # self.rect_g1 = Rectifier()
        # self.rect_g2 = Rectifier()

        # core network
        self.conv_h0 = Convolutional(filter_size=(self.filter_size,self.filter_size),num_filters=self.n_channel, num_channels=self.n_channel ,step=(1,1),border_mode='half',name='conv_h0', **conv_inits)
        # self.conv_h1 = Convolutional(filter_size=(5,5),num_filters=16, num_channels=16,step=(1,1),border_mode='half',name='conv_h1', **conv_inits)
        # self.conv_h2 = Convolutional(filter_size=(5,5),num_filters=16, num_channels=16 ,step=(1,1),border_mode='half',name='conv_h2', **conv_inits)
        self.pool_h0 = MaxPooling(pooling_size=(2,2), name='pool_h0')
        # self.pool_h1 = MaxPooling(pooling_size=(2,2), name='pool_h1')
        # self.pool_h2 = MaxPooling(pooling_size=(2,2), name='pool_h2')
        self.rect_h0 = Rectifier()
        # self.rect_h1 = Rectifier()
        # self.rect_h2 = Rectifier()

        # location network
        linear_l_hidden = [int(self.img_height*self.img_width/16*self.n_channel), 32, self.image_ndim]
        linear_l_activations = [Rectifier(), Logistic()]
        self.linear_l = MLP(activations=linear_l_activations, dims=linear_l_hidden, name="location network", **inits)

        # classification network
        linear_a_hidden = [int(self.img_height*self.img_width/16*self.n_channel), 32, self.n_class]
        linear_a_activations = [Rectifier(), Softmax()]
        self.linear_a = MLP(activations=linear_a_activations, dims=linear_a_hidden, name="classification network", **inits)

        self.flattener = Flattener()

        # self.children = [self.conv_g0, self.conv_g1, self.conv_g2, self.pool_g0, self.pool_g1, self.pool_g2, self.rect_g0, self.rect_g1, self.rect_g2,
        #                  self.conv_h0, self.conv_h1, self.conv_h2, self.pool_h0, self.pool_h1, self.pool_h2,self.rect_h0, self.rect_h1, self.rect_h2,
        #                  self.linear_l, self.linear_a, self.flattener]
        self.children = [self.conv_g0, self.pool_g0, self.rect_g0, self.conv_h0, self.pool_h0, self.rect_h0,
                         self.linear_l, self.linear_a, self.flattener]

    @property
    def output_dim(self):
        return self.n_class

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_g0._push_allocation_config()
        # self.conv_g1._push_allocation_config()
        # self.conv_g2._push_allocation_config()
        self.pool_g0._push_allocation_config()
        # self.pool_g1._push_allocation_config()
        # self.pool_g2._push_allocation_config()
        self.rect_g0._push_allocation_config()
        # self.rect_g1._push_allocation_config()
        # self.rect_g2._push_allocation_config()

        self.conv_h0._push_allocation_config()
        # self.conv_h1._push_allocation_config()
        # self.conv_h2._push_allocation_config()
        self.pool_h0._push_allocation_config()
        # self.pool_h1._push_allocation_config()
        # self.pool_h2._push_allocation_config()
        self.rect_h0._push_allocation_config()
        # self.rect_h1._push_allocation_config()
        # self.rect_h2._push_allocation_config()

        self.linear_l._push_allocation_config()
        self.linear_a._push_allocation_config()

    def get_dim(self, name):
        if name == 'prob':
            return self.n_class # for mnist_lenet
        elif name == 'h0':
            return int(self.img_height * self.img_width * self.n_channel/4) # hieght * width * channel
        # elif name == 'h1':
        #     return 7 *  7 * 16
        # elif name == 'h2':
        #     return 3 * 3 * 16
        elif name == 'l':
            return self.image_ndim
        else:
            super(conv_RAM, self).get_dim(name)

    # ------------------------------------------------------------------------
    @recurrent(sequences=['dummy'], contexts=['x'],
               # states=['l', 'h0', 'h1', 'h2'],
               states=['l', 'h0'],
               # outputs=['l', 'prob', 'h0', 'h1', 'h2', 'out_test'])  # NOTICE: Blocks RNN can only init state in 1D vector !!!
               outputs=['l', 'prob', 'h0', 'rho_orig'])
    # def apply(self, x, dummy, l=None, h0=None, h1=None, h2=None):
    def apply(self, x, dummy, l=None, h0=None):

        h_square = h0.reshape((h0.shape[0], self.n_channel, int(self.img_height/2), int(self.img_width/2)))  # broadcase across channel
        ph0 = self.flattener.apply(self.pool_h0.apply(h_square))
        l = self.linear_l.apply(ph0)
        prob = self.linear_a.apply(ph0)

        if self.image_ndim == 2:
            from theano.tensor.signal.pool import pool_2d
            from attentione2d import ZoomableAttentionWindow

            self.scale = 1
            zoomer_orig = ZoomableAttentionWindow(self.channels, self.img_height, self.img_width, self.read_N, self.scale)
            rho_orig = zoomer_orig.read_small(x, l[:,1]*self.img_height, l[:,0]*self.img_width) # glimpse sensor in 2D, output matrix

            # scale = 2
            # zoomer_larger = ZoomableAttentionWindow(self.channels, self.img_height, self.img_width, self.read_N, scale)
            # rho_larger = zoomer_larger.read_small(x, l[:, 1], l[:, 0])  # glimpse sensor in 2D
            #
            # scale = 4
            # zoomer_largest = ZoomableAttentionWindow(self.channels, self.img_height, self.img_width, self.read_N, scale)
            # rho_largest = zoomer_largest.read_small(x, l[:, 1], l[:, 0])  # glimpse sensor in 2D

        elif self.image_ndim == 3:
            from attentione2d import ZoomableAttentionWindow3d
            zoomer = ZoomableAttentionWindow3d(self.channels, self.img_height, self.img_width, self.img_depth, self.read_N)
            rho = zoomer.read_large(x, l[:,0], l[:,1], l[:,2]) # glimpse sensor in 3D

        # glimpse network
        g0 = self.pool_g0.apply(self.rect_g0.apply(self.conv_g0.apply(rho_orig)))
        # g1 = self.pool_g1.apply(self.rect_g1.apply(self.conv_g1.apply(rho_larger)))
        # g2 = self.pool_g2.apply(self.rect_g2.apply(self.conv_g2.apply(rho_largest)))

        # core network

        # h1 = h1.reshape((g1.shape[0],g1.shape[1],g1.shape[2],g1.shape[3]))
        # h2 = h2.reshape((g2.shape[0],g2.shape[1],g2.shape[2],g2.shape[3]))

        h_square = self.rect_h0.apply(self.conv_h0.apply(g0 + h_square))
        # h1 = self.rect_h1.apply(self.conv_h1.apply(g1 +h1))
        # h2 = self.rect_h2.apply(self.conv_h2.apply(g2 +h2))

        h0 = h_square.reshape((h_square.shape[0],h_square.shape[1]*h_square.shape[2]*h_square.shape[3]))


        # ph1 = self.pool_h0.apply(h1)
        # ph2 = self.pool_h0.apply(h2)
        # h_flatten = T.concatenate([self.flattener.apply(ph0),self.flattener.apply(ph1),self.flattener.apply(ph2)], axis=1)

        # h1 = h1.reshape((h1.shape[0],h1.shape[1]*h1.shape[2]*h1.shape[3]))
        # h2 = h2.reshape((h2.shape[0],h2.shape[1]*h2.shape[2]*h2.shape[3]))
        # return l, prob, h0, h1, h2, tmp
        return l, prob, h0, rho_orig
    # ------------------------------------------------------------------------

    # @application(inputs=['features'], outputs=['l', 'prob', 'h0', 'h1', 'h2', 'out_test'])
    @application(inputs=['features'], outputs=['l', 'prob', 'h0', 'rho_orig'])
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
        #     dtensor5 = T.TensorType(floatX, (False,) * 5)
        #     x = dtensor5(name='input')

        # init_l = tensor.matrix('l')  # for a batch
        # init_h0 = tensor.tensor4('h0')  # for a batch
        # init_h1 = tensor.tensor4('h1')  # for a batch
        # init_h2 = tensor.tensor4('h2')  # for a batch
        # l, prob, h0, h1, h2 = self.apply(x=features, dummy=u, l=init_l, h0=init_h0, h1=init_h1, h2=init_h2 )
        # l, prob, h0, h1, h2, out_test  = self.apply(x=features, dummy=u)
        l, prob, h0, rho_orig = self.apply(x=features, dummy=u)

        # return l, prob, h0, h1, h2, out_test
        return l, prob, h0, rho_orig

if __name__ == "__main__":
    ndim = 2
    # ----------------------------------------------------------------------
    if ndim == 2:
        ram = conv_RAM(image_size=(28,28), channels=1, attention=5, n_iter=3)
    elif ndim==3:
        ram = conv_RAM(image_size=(32,32,32), channels=1, attention=5, n_iter=3)
    ram.push_initialization_config()
    ram.initialize()
    # ------------------------------------------------------------------------
    x = tensor.ftensor4('features')  # keyword from fuel
    y = tensor.matrix('targets')  # keyword from fuel
    l, prob, h0, h1, h2, out_test  = ram.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.

    f = theano.function([x], [l, prob, h0, h1, h2, out_test])
    # test single forward pass
    mnist_train = H5PYDataset('./data/mnist.hdf5', which_sets=('train',))
    handle = mnist_train.open()
    train_data = mnist_train.get_data(handle, slice(0, 1))
    xx = train_data[0]
    print(xx.shape)
    print(train_data[1])
    l, prob, h0, h1, h2, out_test  = f(xx)
    print(out_test)
    print(l)
    print(prob)
    print(h0.shape)
    print(h1.shape)
    print(h2.shape)
