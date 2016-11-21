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
from blocks.bricks.conv import Convolutional, Flattener, ConvolutionalSequence, MaxPooling
from blocks.bricks import Identity, Tanh, Logistic
from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam, Momentum
from blocks.bricks import Tanh, Identity, Softmax, Logistic
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal, Uniform
from blocks.filter import VariableFilter
from blocks.roles import PARAMETER
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from toolz.itertoolz import interleave
from attention import ZoomableAttentionWindow


class DrawClassifyModel(BaseRecurrent, Initializable, Random):
    def __init__(self, image_size, channels, attention, n_iter, **kwargs):
        super(DrawClassifyModel, self).__init__(**kwargs)

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
        n0 = dim_att
        self.rect_linear_g0 = MLP(activations=[Rectifier()], dims=[dim_att*dim_att, n0], name="glimpse network 0", **inits)

        n1 = 2
        self.rect_linear_g1 = MLP(activations=[Rectifier()], dims=[2, n1], name="glimpse network 1", **inits)

        n3 = 10
        self.linear_g21 = MLP(activations=[Identity()], dims=[n0, n3], name="glimpse network 2", **inits)
        self.linear_g22 = MLP(activations=[Identity()], dims=[n1, n3], name="glimpse network 3", **inits)
        self.rect_g = Rectifier()

        # core network
        self.rect_h = Rectifier()
        self.linear_h1 = MLP(activations=[Identity()], dims=[n3, dim_h], name="core network 0", **inits)
        self.linear_h2 = MLP(activations=[Identity()], dims=[n3, dim_h], name="core network 1", **inits)

        # location network
        self.linear_l = MLP(activations=[Identity()], dims=[dim_h, dim_data ], name="location network", **inits)

        # classification network
        self.linear_a = MLP(activations=[Softmax()], dims=[dim_h, n_class], name="classification network", **inits)

        self.children = [self.rect_linear_g0, self.rect_linear_g1, self.linear_g21, self.linear_g22, self.rect_g,
                         self.rect_h, self.linear_h1, self.linear_h2, self.linear_l, self.linear_a]


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

    def get_dim(self, name):
        if name == 'prob':
            return 10 # for mnist_lenet
        elif name == 'h':
            return self.dim_h
        elif name == 'l':
            return self.image_ndim
        else:
            super(DrawClassifyModel, self).get_dim(name)

    # ------------------------------------------------------------------------

    # @recurrent(sequences=['x'], contexts=[],
    #            states=['h'],
    #            outputs=['prob', 'l'])
    @recurrent(sequences=['dummy'], contexts=['x'],
               states=['l', 'h'],
               outputs=['prob', 'l'])
    def apply(self, x, dummy, l=None, h=None):
        if self.image_ndim == 2:
            from attention import ZoomableAttentionWindow
            zoomer = ZoomableAttentionWindow(self.channels, self.img_height, self.img_width, self.read_N)
            rho = zoomer.read_large(x, l[1], l[0]) # glimpse sensor in 2D
        elif self.image_ndim == 3:
            from attention import ZoomableAttentionWindow3d
            zoomer = ZoomableAttentionWindow3d(self.channels, self.img_height, self.img_width, self.img_depth, self.read_N)
            rho = zoomer.read_large(x, l[0], l[1], l[2]) # glimpse sensor in 3D

        h_g = self.rect_linear_g0.apply(rho)  # theta_g^1
        h_l = self.rect_linear_g1.apply(l)  # theta_g^0
        g_t = self.rect_g.apply(self.linear_g21.apply(h_g) + self.linear_g22.apply(h_l))  # theta_g^2
        h = self.rect_h.apply(self.linear_h1.apply(g_t) + self.linear_h2.apply(h_l))
        l = self.linear_l.apply(h)
        prob = self.linear_a.apply(h)

        return l, prob

    # ------------------------------------------------------------------------

    @application(inputs=['features'], outputs=['l', 'prob'])
    def classify(self, features):
        batch_size = features.shape[0]
        # Sample from mean-zeros std.-one Gaussian
        u = self.theano_rng.normal(
            size=(self.n_iter, batch_size, 1),
            avg=0., std=1.)

        if self.image_ndim == 2:
            center_y_ = T.vector()
            center_x_ = T.vector()
            init_l = [center_x_, center_y_]
        else:
            center_x_ = T.vector()
            center_y_ = T.vector()
            center_z_ = T.vector()
            init_l = [center_x_, center_y_, center_z_]
        init_l = tensor.matrix('l')  # for a batch
        l, prob = self.apply(x=features, dummy=u, l=init_l)

        return l, prob

if __name__ == "__main__":
    # x = tensor.vector('x')
    # ram= DrawClassifyModel(image_size=(28,28), channels=1, attention=5)
    # ram.initialize()
    # l, p = ram.apply(inputs=x)
    # f = theano.function([x], [l, p])
    # # l, prob = ram.apply(x=x)
    # for states in f(np.ones((20,1,28,28), dtype=theano.config.floatX)):
    #     print(states)
    #

    # ----------------------------------------------------------------------

    draw = DrawClassifyModel(image_size=(28,28), channels=1, attention=5, n_iter=4)
    draw.push_initialization_config()
    draw.initialize()
    # ------------------------------------------------------------------------
    x = tensor.matrix('features')  # keyword from fuel
    y = tensor.matrix('targets')  # keyword from fuel
    l, prob = draw.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.



    import theano.tensor as tensor
    from blocks.bricks import Rectifier
    dim_att = 5
    n0 = 3
    l = tensor.matrix('l') # for a batch
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    # rho = tensor.lmatrix('targets')
    x = tensor.matrix('features')  # keyword from fuel
    from attention_module.attention import ZoomableAttentionWindow
    zoomer = ZoomableAttentionWindow(1, 28, 28, 5)
    rho = zoomer.read_large(x, l[1], l[0]) # glimpse sensor in 2D
    linear0 = MLP(activations=[Identity()], dims=[dim_att*dim_att, n0], name="glimpse network 0", **inits)
    rect0 = Rectifier()
    h_g = rect0.apply(linear0.apply(rho))  # theta_g^1