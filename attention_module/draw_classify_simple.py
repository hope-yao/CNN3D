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

# from prob_layers import replicate_batch

class Reader(Initializable):
    def __init__(self, x_dim, **kwargs):
        super(Reader, self).__init__(name="reader", **kwargs)

        self.x_dim = x_dim
        self.output_dim = x_dim

    def get_dim(self, name):
        if name == 'input':
            return self.c_dim
        elif name == 'x_dim':
            return self.x_dim
        elif name == 'output':
            return self.output_dim
        else:
            raise ValueError

    @application(inputs=['x'], outputs=['r'])
    def apply(self, x):
        return x

class AttentionReader(Initializable):
    def __init__(self, x_dim, c_dim, channels, height, width, N, **kwargs):
        super(AttentionReader, self).__init__(name="reader", **kwargs)

        self.img_height = height
        self.img_width = width
        self.N = N
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.output_dim = (height, width)

        self.zoomer = ZoomableAttentionWindow(channels, height, width, N)
        self.readout = MLP(activations=[Identity()], dims=[c_dim, 2],
                           **kwargs)  # input is the output from RNN
        # reader_dim = [c_dim, 16, 2]
        # self.readout = MLP(activations=[Rectifier(), Rectifier()], dims=reader_dim, **kwargs) # input is the output from RNN

        self.children = [self.readout]

    def get_dim(self, name):
        if name == 'input':
            return self.c_dim
        elif name == 'x_dim':
            return self.x_dim
        elif name == 'output':
            return self.output_dim
        else:
            raise ValueError

    @application(inputs=['x', 'c'], outputs=['r', 'cx', 'cy'])
    def apply(self, x, c):
        l = self.readout.apply(c)

        center_y, center_x = self.zoomer.nn2att_const_gamma(l)

        r = self.zoomer.read_large(x, center_y, center_x)

        return r, center_x, center_y

class DrawClassifyModel(BaseRecurrent, Initializable, Random):
    def __init__(self, image_size, channels, attention, **kwargs):
        self.channels = channels
        self.read_N = attention
        self.image_ndim = len(image_size)
        if self.image_ndim == 2:
            self.img_height, self.img_width = image_size
        elif self.image_ndim == 3:
            self.img_height, self.img_width, self.img_depth = image_size
        super(DrawClassifyModel, self).__init__(**kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    # def _push_allocation_config(self):
    #     # self.reader._push_allocation_config()
    #     self.conv_sequence._push_allocation_config()
    #     conv_out_dim = self.conv_sequence.get_dim('output')
    #
    #     self.top_mlp.activations = self.top_mlp_activations
    #     self.top_mlp.dims = [np.prod(conv_out_dim)] + self.top_mlp_dims


    def get_dim(self, name):
        if name == 'y':
            return 10 # for mnist_lenet
        elif name == 'c':
            return self.conv_out_dim_flatten
        elif name == 'r':
            return self.x_dim
        elif name == 'center_y':
            return 1
        elif name == 'center_x':
            return 1
        elif name == 'delta':
            return 1
        else:
            super(DrawClassifyModel, self).get_dim(name)

    # ------------------------------------------------------------------------

    @recurrent(sequences=['x'], contexts=[],
               states=['h'],
               outputs=['prob', 'l'])
    def apply(self, x, l, h):
        dim_att = 28
        n_class = 10
        dim_h = 100
        dim_data = 2
        inits = {
            'weights_init': IsotropicGaussian(0.01),
            'biases_init': Constant(0.),
        }

        if self.image_ndim == 2:
            from attention import ZoomableAttentionWindow
            zoomer = ZoomableAttentionWindow(self.channels, self.img_height, self.img_width, self.read_N)
            rho = zoomer.read_large(x, l[1], l[0]) # glimpse sensor in 2D
        elif self.image_ndim == 3:
            from attention import ZoomableAttentionWindow3d
            zoomer = ZoomableAttentionWindow3d(self.channels, self.img_height, self.img_width, self.img_depth, self.read_N)
            rho = zoomer.read_large(x, l[0], l[1], l[2]) # glimpse sensor in 2D

        # glimpse network
        n0 = dim_att
        linear0 = MLP(activations=[Identity()], dims=[dim_att*dim_att, n0], name="glimpse network 0", **inits)
        rect0 = Rectifier()
        h_g = rect0.apply(linear0.apply(rho))  # theta_g^1

        n1 = 2
        linear1 = MLP(activations=[Identity()], dims=[2, n1], name="glimpse network 1", **inits)
        rect1 = Rectifier()
        h_l = rect1.apply(linear1.apply(l))  # theta_g^0

        n3 = 10
        linear3 = MLP(activations=[Identity()], dims=[n0, n3], name="glimpse network 2", **inits)
        linear4 = MLP(activations=[Identity()], dims=[n1, n3], name="glimpse network 3", **inits)
        rect3 = Rectifier()
        g_t = rect3.apply(linear3.apply(h_l) + linear4.apply(h_g))  # theta_g^2

        # core network
        rect4 = Rectifier()
        linear5 = MLP(activations=[Identity()], dims=[n3, dim_h], name="core network 0", **inits)
        linear6 = MLP(activations=[Identity()], dims=[n3, dim_h], name="core network 1", **inits)
        h = rect4.apply(linear5.apply(h) + linear6.apply(g_t))

        # location network
        l = MLP(activations=[Identity()], dims=[dim_h, dim_data ], name="location network", **inits)

        # classification network
        prob = MLP(activations=[Softmax()], dims=[dim_h, n_class], name="classification network", **inits)
        return l, prob

    # ------------------------------------------------------------------------

    @application(inputs=['features'], outputs=['targets', 'r', 'c', 'cx', 'cy'])
    def classify(self, features):

        l, prob = self.apply(x=features)

        return l, prob
