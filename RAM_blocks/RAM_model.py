# MAX: simple RNN model w/o VAE


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
from bricks3D.cnn3d_bricks import Convolutional3, MaxPooling3, ConvolutionalSequence3, Flattener3


class RAM(BaseRecurrent, Initializable, Random):
    def __init__(self, image_size, channels, attention, n_iter, n_class, **kwargs):
        super(RAM, self).__init__(**kwargs)

        self.n_iter = n_iter
        self.channels = channels
        self.read_N = attention
        self.image_ndim = len(image_size)
        if self.image_ndim == 2:
            self.img_height, self.img_width = image_size
        elif self.image_ndim == 3:
            self.img_height, self.img_width, self.img_depth = image_size
        self.dim_h = 8

        ########change according to number of classes
        self.n_class = n_class

        dim_h = self.dim_h
        inits = {
            # 'weights_init': Constant(1.),
            # 'biases_init': Constant(0.),
            'weights_init': Orthogonal(),
            'biases_init': IsotropicGaussian(),
        }

        # glimpse network
        n0 = 8
        self.rect_linear_g0 = MLP(activations=[Rectifier()], dims=[3*self.read_N**self.image_ndim, n0], name="glimpse network 0", **inits) # 3 glimpse of different resolution

        n1 = 8
        self.rect_linear_g1 = MLP(activations=[Rectifier()], dims=[self.image_ndim, n1], name="glimpse network 1", **inits)

        self.linear_g21 = MLP(activations=[Identity()], dims=[n0, dim_h], name="glimpse network 2", **inits)
        self.linear_g22 = MLP(activations=[Identity()], dims=[n1, dim_h], name="glimpse network 3", **inits)
        self.rect_g = Rectifier()

        # core network
        self.rect_h = Rectifier()
        self.linear_h1 = MLP(activations=[Identity()], dims=[dim_h, dim_h], name="core network 0", **inits)
        self.linear_h2 = MLP(activations=[Identity()], dims=[dim_h, dim_h], name="core network 1", **inits)

        # location network
        self.linear_l = MLP(activations=[Logistic()], dims=[dim_h, self.image_ndim], name="location network", **inits)

        # classification network
        self.linear_a = MLP(activations=[Softmax()], dims=[dim_h, self.n_class], name="classification network", **inits)

        self.scale1 = 1
        self.scale2 = 1
        self.pool_3d_1 = MaxPooling3((self.scale1, self.scale1, self.scale1))
        self.pool_3d_2 = MaxPooling3((self.scale2, self.scale2, self.scale2))
        self.children = [self.rect_linear_g0, self.rect_linear_g1, self.linear_g21, self.linear_g22, self.rect_g,
                         self.rect_h, self.linear_h1, self.linear_h2, self.linear_l, self.linear_a, self.pool_3d_1, self.pool_3d_2]

        # self.children = [self.rect_linear_g0, self.rect_linear_g1, self.linear_g21, self.linear_g22, self.rect_g,
        #                  self.rect_h, self.linear_h1, self.linear_h2, self.linear_l, self.linear_a]

    @property
    def output_dim(self):
        return self.n_class

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.rect_linear_g0._push_allocation_config()
        self.rect_linear_g1._push_allocation_config()
        self.linear_g21._push_allocation_config()
        self.linear_g22._push_allocation_config()
        self.rect_g._push_allocation_config()
        self.rect_h._push_allocation_config()
        self.linear_h1._push_allocation_config()
        self.linear_h2._push_allocation_config()
        self.linear_l._push_allocation_config()
        self.linear_a._push_allocation_config()
        # self.fork._push_allocation_config()
        # self.softmax._push_allocation_config()

    def get_dim(self, name):
        if name == 'prob':
            return self.n_class
        elif name == 'h':
            return self.dim_h
        elif name == 'l':
            return self.image_ndim
        else:
            super(RAM, self).get_dim(name)

    # ------------------------------------------------------------------------
    @recurrent(sequences=['dummy'], contexts=['x'],
               states=['l', 'h'],
               outputs=['l', 'prob', 'h', 'rho_orig', 'rho_larger', 'rho_largest'])  # h seems not necessary
               # outputs=['l', 'prob', 'h', 'rho_orig'])
    def apply(self, x, dummy, l=None, h=None):
        if self.image_ndim == 2:
            from theano.tensor.signal.pool import pool_2d
            from attentione2d import ZoomableAttentionWindow

            scale = 1
            zoomer_orig = ZoomableAttentionWindow(self.channels, self.img_height, self.img_width, self.read_N, scale )
            rho_orig = zoomer_orig.read_patch(x, l[:,1]*28, l[:,0]*28) # glimpse sensor in 2D
            rho_orig = rho_orig.reshape((x.shape[0], self.channels*self.read_N*self.read_N))

            scale = self.scale1
            N_larger = scale *self.read_N
            zoomer_larger = ZoomableAttentionWindow(self.channels, self.img_height, self.img_width, N_larger, scale ) #accurally the last parameter is not used in read_patch function
            rho_larger = zoomer_larger.read_patch(x, l[:, 1]*28, l[:, 0]*28)  # glimpse sensor in 2D
            rho_larger = pool_2d(rho_larger,(scale ,scale )) # downsampling
            rho_larger = rho_larger.reshape((rho_larger.shape[0], self.channels*self.read_N*self.read_N))

            scale = self.scale2
            N_larger = scale *self.read_N
            zoomer_largest = ZoomableAttentionWindow(self.channels, self.img_height, self.img_width, N_larger, scale )
            rho_largest = zoomer_largest.read_patch(x, l[:, 1]*28, l[:, 0]*28)  # glimpse sensor in 2D
            rho_largest = pool_2d(rho_largest,(scale ,scale )) # downsampling
            rho_largest = rho_largest.reshape((rho_largest.shape[0], self.channels*self.read_N*self.read_N))

            rho = T.concatenate([rho_orig, rho_larger, rho_largest], axis=1)

        elif self.image_ndim == 3:
            from attentione3d import ZoomableAttentionWindow3d
            shiftx = shifty = shiftz = 0.0  # change initial attention window to the center
            dim_size = 32
            zoomer_orig = ZoomableAttentionWindow3d(self.channels, self.img_height, self.img_width, self.img_depth, self.read_N, 1)
            rho_orig = zoomer_orig.read_small(x, (l[:, 0]+shiftx)*dim_size, (l[:, 1]+shifty)*dim_size, (l[:, 2]+shiftz)*dim_size)  # glimpse sensor in 2D
            rho_orig = rho_orig.reshape((x.shape[0], self.channels * self.read_N * self.read_N * self.read_N))

            # zoomer_larger = ZoomableAttentionWindow3d(self.channels, self.img_height, self.img_width, self.img_depth,  self.read_N, self.scale1)
            # rho_larger = zoomer_larger.read_small(x, (l[:, 0]+shiftx)*dim_size, (l[:, 1]+shifty)*dim_size, (l[:, 2]+shiftz)*dim_size)  # glimpse sensor in 2D
            # rho_larger = self.pool_3d_1.apply(rho_larger)  # downsampling
            # rho_larger = rho_larger.reshape((rho_larger.shape[0], self.channels * self.read_N * self.read_N * self.read_N))
            #
            # zoomer_largest = ZoomableAttentionWindow3d(self.channels, self.img_height, self.img_width, self.img_depth,  self.read_N, self.scale1)
            #
            # rho_largest = zoomer_largest.read_small(x, (l[:, 0]+shiftx)*dim_size, (l[:, 1]+shifty)*dim_size, (l[:, 2]+shiftz)*dim_size)  # glimpse sensor in 2D
            # rho_largest = self.pool_3d_2.apply(rho_largest)  # downsampling
            # rho_largest = rho_largest.reshape((rho_largest.shape[0], self.channels * self.read_N * self.read_N * self.read_N))

            # rho = T.concatenate([rho_orig, rho_larger, rho_largest], axis=1)
            rho = rho_orig

        h_g = self.rect_linear_g0.apply(rho)  # theta_g^0
        h_l = self.rect_linear_g1.apply(l)  # theta_g^1
        g_t = self.rect_g.apply(self.linear_g21.apply(h_g) + self.linear_g22.apply(h_l))  # theta_g^2
        h = self.rect_h.apply(self.linear_h1.apply(g_t) + self.linear_h2.apply(h))
        l = self.linear_l.apply(0.1*h)
        prob = self.linear_a.apply(h)
        # l, _prob = self.fork.apply(h)
        # prob = self.softmax.apply(_prob)
        return l, prob, h, rho_orig, rho_larger, rho_largest
        # return l, prob, h, rho_orig

    # ------------------------------------------------------------------------

    @application(inputs=['features'], outputs=['l', 'prob', 'rho_orig', 'rho_larger', 'rho_largest'])
    # @application(inputs=['features'], outputs=['l', 'prob', 'rho_orig'])
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
        l, prob, h, rho_orig, rho_larger, rho_largest = self.apply(x=features, dummy=u)
        # l, prob, h, rho_orig = self.apply(x=features, dummy=u)
        return l, prob, rho_orig, rho_larger, rho_largest
        # return l, prob, rho_orig

if __name__ == "__main__":
    ndim = 3
    # ----------------------------------------------------------------------
    if ndim==2:
        ram = RAM(image_size=(28,28), channels=1, attention=5, n_iter=4)
        ram.push_initialization_config()
        ram.initialize()
        # ------------------------------------------------------------------------
        x = tensor.ftensor4('features')  # keyword from fuel
        y = tensor.matrix('targets')  # keyword from fuel
        l, prob, rho_orig, rho_larger, rho_largest = ram.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.
        f = theano.function([x], [l, prob, rho_orig, rho_larger, rho_largest])
        # ------------------------------------------------------------------------
        # test single forward pass
        from fuel.datasets.hdf5 import H5PYDataset
        mnist_train = H5PYDataset('../data/mnist.hdf5', which_sets=('train',))
        handle = mnist_train.open()
        train_data = mnist_train.get_data(handle, slice(0, 16))
        xx = train_data[0]
        print(xx.shape)
        l, prob, rho_orig, rho_larger, rho_largest = f(xx)
        print(l)
        print(prob)
    elif ndim==3:
        ram = RAM(image_size=(32,32,32), channels=1, attention=7, n_iter=4)
        ram.push_initialization_config()
        ram.initialize()
        # ------------------------------------------------------------------------
        dtensor5 = T.TensorType('float32', (False,) * 5)
        x = dtensor5('features')  # keyword from fuel
        y = tensor.matrix('targets')  # keyword from fuel
        l, prob, rho_orig, rho_larger, rho_largest = ram.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.
        f = theano.function([x], [l, prob, rho_orig, rho_larger, rho_largest])
        # ------------------------------------------------------------------------
        # test single forward pass
        from fuel.datasets.hdf5 import H5PYDataset
        mnist_train = H5PYDataset('../data/shapenet10.hdf5', which_sets=('train',))
        handle = mnist_train.open()
        train_data = mnist_train.get_data(handle, slice(0, 16))
        xx = train_data[0]
        print(xx.shape)
        l, prob, rho_orig, rho_larger, rho_largest = f(xx)
        print(l)
        print(prob)