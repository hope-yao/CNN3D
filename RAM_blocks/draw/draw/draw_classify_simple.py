# MAX: simple RNN model w/o VAE


from __future__ import division, print_function

import sys

sys.path.append("../lib")

import theano.tensor as T
import numpy as np

from blocks.bricks.base import application
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Random, Initializable, MLP, Rectifier
from blocks.bricks.conv import Convolutional, Flattener, ConvolutionalSequence, MaxPooling
from blocks.bricks import Softmax
from blocks.initialization import Constant, IsotropicGaussian, Uniform
from toolz.itertoolz import interleave

from RAM_blocks.draw.draw.attention import ZoomableAttentionWindow


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
        # self.readout = MLP(activations=[Identity()], dims=[c_dim, 2],
        #                    **kwargs)  # input is the output from RNN
        reader_dim = [c_dim, 16, 2]
        self.readout = MLP(activations=[Rectifier(), Rectifier()], dims=reader_dim, **kwargs) # input is the output from RNN

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
        super(DrawClassifyModel, self).__init__(**kwargs)

        self.n_iter = 3
        y_dim = 10
        rnn_dim = 64
        num_filters = 16

        rnninits = {
            # 'weights_init': Orthogonal(),
            'weights_init': IsotropicGaussian(0.01),
            'biases_init': Constant(0.),
        }
        inits = {
            # 'weights_init': Orthogonal(),
            'weights_init': IsotropicGaussian(0.01),
            'biases_init': Constant(0.),
        }
        conv_inits = {
            'weights_init': Uniform(width=.2),
            'biases_init': Constant(0.)
        }
        img_height, img_width = image_size
        self.x_dim = channels * img_height * img_width

        # # Configure attention mechanism
        # read_N = attention
        # read_N = int(read_N)
        # read_dim = x_dim
        #
        # reader = AttentionReader(x_dim=x_dim, c_dim=rnn_dim,
        #                          channels=channels, width=img_width, height=img_height,
        #                          N=read_N, **inits)

        # encoder_conv = Convolutional(filter_size=(read_N, read_N), num_filters=num_filters, num_channels=channels, name="CONV_enc", **inits)
        # conv_dim = (read_N-read_N+1)**2*num_filters + 4 # cx, cy, delta, sigma
        # encoder_mlp = MLP([Identity()], [conv_dim, 4 * rnn_dim], name="MLP_enc", **inits)
        # rnn = LSTM(dim=rnn_dim, name="RNN", **rnninits)
        # decoder_mlp = MLP([Softmax()], [rnn_dim, y_dim], name="MLP_dec", **inits)

        # self.reader = reader
        # self.encoder_conv = encoder_conv
        # self.encoder_mlp = encoder_mlp
        # self.rnn = rnn
        # self.decoder_mlp = decoder_mlp

        # self.children = [self.reader, self.encoder_conv, self.encoder_mlp, self.rnn,
        #                  self.decoder_mlp]


#-----------------------------------------------------------------------------------------------------------------------
        # USE LeNet

        feature_maps = [20, 50] #[20, 50]
        mlp_hiddens = [500] # 500
        conv_sizes = [5, 5] # [5, 5]
        pool_sizes = [2, 2]
        image_size = (28, 28)
        output_size = 10

        conv_activations = [Rectifier() for _ in feature_maps]
        mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]

        num_channels = 1
        image_shape = (28, 28)
        filter_sizes = zip(conv_sizes, conv_sizes)
        feature_maps = feature_maps
        pooling_sizes = zip(pool_sizes, pool_sizes)
        top_mlp_activations = mlp_activations
        top_mlp_dims = mlp_hiddens + [output_size]
        border_mode = 'full'

        conv_step = None
        if conv_step is None:
            self.conv_step = (1, 1)
        else:
            self.conv_step = conv_step
        self.num_channels = num_channels
        self.image_shape = image_shape
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims
        self.border_mode = border_mode

        conv_parameters = zip(filter_sizes, feature_maps)

        # Construct convolutional layers with corresponding parameters
        self.layers = list(interleave([
            (Convolutional(filter_size=filter_size,
                           num_filters=num_filter,
                           step=self.conv_step,
                           border_mode=self.border_mode,
                           name='conv_{}'.format(i))
             for i, (filter_size, num_filter)
             in enumerate(conv_parameters)),
            conv_activations,
            (MaxPooling(size, name='pool_{}'.format(i))
             for i, size in enumerate(pooling_sizes))]))

        self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
                                                   image_size=image_shape, **conv_inits)

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims, **inits)
        self.flattener = Flattener()

# -----------------------------------------------------------------------------------------------------------------------
        # Configure attention mechanism
        read_N = attention
        read_N = int(read_N)

        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')
        self.conv_out_dim_flatten = np.prod(conv_out_dim)
        reader = AttentionReader(x_dim=self.x_dim, c_dim=self.conv_out_dim_flatten,
                                 channels=channels, width=img_width, height=img_height,
                                 N=read_N, **inits)
        # reader = Reader(x_dim=self.x_dim)

        self.reader = reader

        self.children = [self.reader, self.conv_sequence, self.flattener, self.top_mlp]

        # application_methods = [self.reader.apply, self.conv_sequence.apply, self.flattener.apply,
        #                        self.top_mlp.apply]
        # super(DrawClassifyModel, self).__init__(application_methods, **conv_inits)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        # self.reader._push_allocation_config()
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [np.prod(conv_out_dim)] + self.top_mlp_dims


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

    @recurrent(sequences=['dummy'], contexts=['x'],
               states=['r', 'c'],
               outputs=['y', 'r', 'c', 'cx', 'cy'])
    def apply(self, c, r, x, dummy):
        # r, cx, cy, delta, sigma = self.reader.apply(x, c)
        # a = self.encoder_conv.apply(r)
        # # a_flatten = Flattener(a)
        # # aa = T.flatten(a,outdim=2)
        # aa = T.concatenate([T.flatten(a,outdim=2),  T.stack([cx, cy, delta, sigma]).T], axis=1)
        # i = self.encoder_mlp.apply(aa)
        # h, cc = self.rnn.apply(states=h, cells=c, inputs=i,
        #                       iterate=False)
        # c = c + cc
        # y = self.decoder_mlp.apply(c)

        rr, center_y, center_x = self.reader.apply(x, c)
        r = T.minimum(r + rr,1.) # combine revealed images
        batch_size = r.shape[0]
        c_raw = self.conv_sequence.apply(r.reshape((batch_size,1,28,28)))
        c = self.flattener.apply(c_raw)
        y = self.top_mlp.apply(c)

        return y, r, c, center_x, center_y

    # ------------------------------------------------------------------------

    @application(inputs=['features'], outputs=['targets', 'r', 'c', 'cx', 'cy'])
    def classify(self, features):
        batch_size = features.shape[0]
        # Sample from mean-zeros std.-one Gaussian
        u = self.theano_rng.normal(
            size=(self.n_iter, batch_size, 1),
            avg=0., std=1.)

        # y, r, c, center_x, center_y, delta, sigma = self.apply(x=features, dummy=u)
        y, r, c, cx, cy = self.apply(x=features, dummy=u)

        return y, r, c, cx, cy
