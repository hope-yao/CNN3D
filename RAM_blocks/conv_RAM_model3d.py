#!/usr/bin/env python


from __future__ import division, print_function

import sys

sys.path.append("../lib")
from cnn3d_bricks import Convolutional3, MaxPooling3, ConvolutionalSequence3, Flattener3

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
        self.channels = channels
        self.read_N = attention
        self.image_ndim = len(image_size)
        self.img_height, self.img_width, self.img_depth = image_size
        self.n_class = n_class

        dim_data = 3
        inits = {
            # 'weights_init': IsotropicGaussian(0.01),
            'biases_init': Constant(0.),
            'weights_init': Orthogonal(),
            # 'biases_init': IsotropicGaussian(),
        }
        conv_inits = {
            'weights_init': Uniform(width=.2),
            'biases_init': Constant(0.)
        }

        # glimpse network
        conv_g0 = Convolutional3(filter_size=(5,5,5),num_filters=1, num_channels=1 ,step=(1,1,1),border_mode='half',name='conv_g0', **conv_inits)
        pool_g0 = MaxPooling3(pooling_size=(8,8,8), name='pool_g0')
        rect_g0 = Rectifier()
        self.layers = list([conv_g0, rect_g0, pool_g0])
        self.conv_sequence = ConvolutionalSequence3(self.layers, 1,
                                                   image_size=image_size, **conv_inits)

        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')
        self.conv_out_dim_flatten = np.prod(conv_out_dim)

        # location network
        self.linear_l = MLP(activations=[Logistic()], dims=[self.conv_out_dim_flatten, self.image_ndim], name="location network", **inits)

        # classification network
        self.linear_a = MLP(activations=[Softmax()], dims=[self.conv_out_dim_flatten, self.n_class], name="classification network", **inits)

        self.flattener = Flattener3()

        self.children = [self.conv_sequence, self.flattener, self.linear_l, self.linear_a]

    @property
    def output_dim(self):
        return self.n_class

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_sequence._push_allocation_config()
        self.linear_l._push_allocation_config()
        self.linear_a._push_allocation_config()

    def get_dim(self, name):
        if name == 'prob':
            return self.n_class # for mnist_lenet
        elif name == 'l':
            return self.image_ndim
        elif name == 'c':
            return self.img_height*self.img_width*self.img_depth
        else:
            super(conv_RAM, self).get_dim(name)

    # ------------------------------------------------------------------------
    @recurrent(sequences=['dummy'], contexts=['x'],
               states=['l', 'c'],
               outputs=['l', 'prob', 'c'])  # NOTICE: Blocks RNN can only init state in 1D vector !!!
    def apply(self, x, dummy, l=None, c=None):
        from attentione3d import ZoomableAttentionWindow3d
        scale = 1
        zoomer = ZoomableAttentionWindow3d(self.channels, self.img_height, self.img_width, self.img_depth, self.read_N, scale)
        rho = zoomer.read_large(x, l[:,0]*32+16, l[:,1]*32+6, l[:,2]*32+16) # glimpse sensor in 3D
        c = c + rho

        # glimpse network
        h = self.conv_sequence.apply(c.reshape((c.shape[0], self.channels, self.img_height, self.img_width, self.img_depth)))

        h_flatten = self.flattener.apply(h)
        prob = self.linear_a.apply(h_flatten)
        l = self.linear_l.apply(h_flatten)

        return l, prob, c

    # ------------------------------------------------------------------------

    @application(inputs=['features'], outputs=['l', 'prob', 'c'])
    def classify(self, features):
        batch_size = features.shape[0]
        # No particular use apart from control n_steps in RNN
        u = self.theano_rng.normal(
            size=(self.n_iter, batch_size, 1),
            avg=0., std=1.)

        l, prob, c = self.apply(x=features, dummy=u)

        return l, prob, c

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
    l, prob, c = ram.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.

    f = theano.function([x], [l, prob, c])
    # test single forward pass
    mnist_train = H5PYDataset('./data/mnist.hdf5', which_sets=('train',))
    handle = mnist_train.open()
    train_data = mnist_train.get_data(handle, slice(0, 1))
    xx = train_data[0]
    print(xx.shape)
    print(train_data[1])
    l, prob, c = f(xx)
    print(l)
    print(prob)
    print(c.shape)

