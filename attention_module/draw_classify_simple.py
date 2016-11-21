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


class RAM(BaseRecurrent, Initializable, Random):
    def __init__(self, image_size, channels, attention, n_iter, **kwargs):
        super(RAM, self).__init__(**kwargs)

        self.n_iter = n_iter
        self.channels = channels
        self.read_N = attention
        self.image_ndim = len(image_size)
        if self.image_ndim == 2:
            self.img_height, self.img_width = image_size
        elif self.image_ndim == 3:
            self.img_height, self.img_width, self.img_depth = image_size
        self.dim_h = 100

        l = tensor.matrix('l')  # for a batch
        dim_att = 28
        n_class = 10
        dim_h = self.dim_h
        dim_data = 2
        inits = {
            'weights_init': IsotropicGaussian(0.01),
            'biases_init': Constant(0.),
        }

        # glimpse network
        n0 = 50
        self.rect_linear_g0 = MLP(activations=[Rectifier()], dims=[dim_att*dim_att, n0], name="glimpse network 0", **inits)

        n1 = 20
        self.rect_linear_g1 = MLP(activations=[Rectifier()], dims=[2, n1], name="glimpse network 1", **inits)

        self.linear_g21 = MLP(activations=[Identity()], dims=[n0, dim_h], name="glimpse network 2", **inits)
        self.linear_g22 = MLP(activations=[Identity()], dims=[n1, dim_h], name="glimpse network 3", **inits)
        self.rect_g = Rectifier()

        # core network
        self.rect_h = Rectifier()
        self.linear_h1 = MLP(activations=[Identity()], dims=[dim_h, dim_h], name="core network 0", **inits)
        self.linear_h2 = MLP(activations=[Identity()], dims=[dim_h, dim_h], name="core network 1", **inits)

        # location network
        self.linear_l = MLP(activations=[Identity()], dims=[dim_h, dim_data], name="location network", **inits)

        # classification network
        self.linear_a = MLP(activations=[Softmax()], dims=[dim_h, n_class], name="classification network", **inits)

        self.children = [self.rect_linear_g0, self.rect_linear_g1, self.linear_g21, self.linear_g22, self.rect_g,
                         self.rect_h, self.linear_h1, self.linear_h2, self.linear_l, self.linear_a]

        # self.fork = Fork(prototype = Linear(use_bias=True),
        #                          output_names = ['l', 'a'], input_dim = dim_h, output_dims = [dim_data, n_class],
        #                          name="location classification", **inits)
        # self.softmax = Softmax(name="softmax")
        #
        # self.children = [self.rect_linear_g0, self.rect_linear_g1, self.linear_g21, self.linear_g22, self.rect_g,
        #                  self.rect_h, self.linear_h1, self.linear_h2, self.fork, self.softmax]
    @property
    def output_dim(self):
        return 10

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
            return 10 # for mnist_lenet
        elif name == 'h':
            return self.dim_h
        elif name == 'l':
            return self.image_ndim
        else:
            super(RAM, self).get_dim(name)

    # ------------------------------------------------------------------------

    @recurrent(sequences=['dummy'], contexts=['x'],
               states=['l', 'h'],
               outputs=['l', 'prob', 'h'])  # h seems not necessary
    def apply(self, x, dummy, l=None, h=None):
        if self.image_ndim == 2:

            from attention import ZoomableAttentionWindow
            zoomer = ZoomableAttentionWindow(self.channels, self.img_height, self.img_width, self.read_N)
            rho = zoomer.read_large(x, l[:,1], l[:,0]) # glimpse sensor in 2D
        elif self.image_ndim == 3:
            from attention import ZoomableAttentionWindow3d
            zoomer = ZoomableAttentionWindow3d(self.channels, self.img_height, self.img_width, self.img_depth, self.read_N)
            rho = zoomer.read_large(x, l[:,0], l[:,1], l[:,2]) # glimpse sensor in 3D

        h_g = self.rect_linear_g0.apply(rho)  # theta_g^0
        h_l = self.rect_linear_g1.apply(l)  # theta_g^1
        g_t = self.rect_g.apply(self.linear_g21.apply(h_g) + self.linear_g22.apply(h_l))  # theta_g^2
        h = self.rect_h.apply(self.linear_h1.apply(g_t) + self.linear_h2.apply(h))
        l = self.linear_l.apply(h)
        prob = self.linear_a.apply(h)
        # l, _prob = self.fork.apply(h)
        # prob = self.softmax.apply(_prob)
        return l, prob, h

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
        if self.image_ndim == 2:
            center_y_ = T.ones((bs,)) * img/2
            center_x_ = T.ones((bs,)) * img/2
            init_l = [center_x_, center_y_]
        else:
            center_x_ = T.vector()
            center_y_ = T.vector()
            center_z_ = T.vector()
            init_l = [center_x_, center_y_, center_z_]
        # init_l = tensor.matrix('l')  # for a batch
        l, prob, h = self.apply(x=features, dummy=u)

        return l, prob

if __name__ == "__main__":

    # ----------------------------------------------------------------------

    ram = RAM(image_size=(28,28), channels=1, attention=5, n_iter=4)
    ram.push_initialization_config()
    ram.initialize()
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
    print(prob)