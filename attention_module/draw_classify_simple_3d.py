# MAX: simple RNN model w/o VAE


from __future__ import division, print_function

import sys

sys.path.append("../lib")

import numpy as np

from blocks.bricks.base import application
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Random, Initializable, MLP, Rectifier
from blocks.bricks import Identity, Softmax
from blocks.initialization import Constant, IsotropicGaussian, Uniform
from toolz.itertoolz import interleave
from bricks3D.cnn3d_bricks import Convolutional3, MaxPooling3, ConvolutionalSequence3, Flattener3

from attention import ZoomableAttentionWindow3d
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

class AttentionReader3d(Initializable):
    def __init__(self, x_dim, c_dim, channels, height, width, depth, N, **kwargs):
        super(AttentionReader3d, self).__init__(name="reader", **kwargs)

        self.img_height = height
        self.img_width = width
        self.img_depth = depth
        self.N = N
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.output_dim = (height, width, depth)

        self.zoomer = ZoomableAttentionWindow3d(channels, height, width, depth, N)
        self.readout = MLP(activations=[Identity()], dims=[c_dim, 3], **kwargs) # input is the output from RNN

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

    @application(inputs=['x', 'c'], outputs=['r', 'cx', 'cy', 'cz'])
    def apply(self, x, c):
        l = self.readout.apply(c)

        center_x, center_y , center_z = self.zoomer.nn2att_const_gamma(l)

        r = self.zoomer.read_large(x, center_x, center_y, center_z)

        return r, center_x, center_y, center_z

class DrawClassifyModel3d(BaseRecurrent, Initializable, Random):
    def __init__(self, image_size, channels, attention, **kwargs):
        super(DrawClassifyModel3d, self).__init__(**kwargs)

        self.n_iter = 1

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
        img_height, img_width, img_depth  = image_size
        self.x_dim = channels * img_height * img_width * img_depth

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

        feature_maps = [16, 32] #[20, 50]
        mlp_hiddens = [500] # 500
        conv_sizes = [5, 5, 5] # [5, 5]
        pool_sizes = [2, 2, 2]
        # image_size = (28, 28)
        output_size = 10

        conv_activations = [Rectifier() for _ in feature_maps]
        mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]

        num_channels = 1
        image_shape = (32, 32, 32)
        filter_sizes = [(5,5,5),(5,5,5)]
        feature_maps = feature_maps
        pooling_sizes = [(2,2,2),(2,2,2)]
        top_mlp_activations = mlp_activations
        top_mlp_dims = mlp_hiddens + [output_size]
        border_mode = 'valid'

        conv_step = None
        if conv_step is None:
            self.conv_step = (1, 1, 1)
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
            (Convolutional3(filter_size=filter_size,
                           num_filters=num_filter,
                           step=self.conv_step,
                           border_mode=self.border_mode,
                           name='conv_{}'.format(i))
             for i, (filter_size, num_filter)
             in enumerate(conv_parameters)),
            conv_activations,
            (MaxPooling3(size, name='pool_{}'.format(i))
             for i, size in enumerate(pooling_sizes))]))

        self.conv_sequence = ConvolutionalSequence3(self.layers, num_channels,
                                                   image_size=image_shape, **conv_inits)

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims, **inits)
        self.flattener = Flattener3()

# -----------------------------------------------------------------------------------------------------------------------
        # Configure attention mechanism
        read_N = attention
        read_N = int(read_N)

        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')
        self.conv_out_dim_flatten = np.prod(conv_out_dim)
        # reader = AttentionReader3d(x_dim=self.x_dim, c_dim=self.conv_out_dim_flatten,
        #                          channels=channels, width=img_width, height=img_height, depth=img_depth,
        #                          N=read_N, **inits)
        reader = Reader(x_dim=self.x_dim)
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
        elif name == 'center_z':
            return 1
        elif name == 'delta':
            return 1
        else:
            super(DrawClassifyModel3d, self).get_dim(name)

    # ------------------------------------------------------------------------

    @recurrent(sequences=['dummy'], contexts=['x'],
               states=['r', 'c'],
               outputs=['y', 'r', 'c'])
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

        # rr, center_x, center_y, center_z = self.reader.apply(x, c)
        rr = self.reader.apply(x)
        r = r + rr # combine revealed images
        batch_size = r.shape[0]
        c_raw = self.conv_sequence.apply(r.reshape((batch_size,1,32,32,32)))
        c = self.flattener.apply(c_raw)
        y = self.top_mlp.apply(c)

        return y, r, c

    # @recurrent(sequences=['dummy'], contexts=['x'],
    #            states=['r', 'l'],
    #            outputs=['y', 'r', 'l', 'cx', 'cy'])
    #
    # '''l is location of attention,
    #    h is the out put of
    # '''
    # def apply(self, l, r, x, dummy):
    #     # glimpse network
    #     rho = self.reader.apply(x, l) # glimpse sensor
    #     h_g = rect_linear0(rho)  # theta_g^1
    #     h_l = rect_linear1(l)  # theta_g^0
    #     g_t = rect(linear(h_l) + linear(h_g))  # theta_g^2
    #
    #     # core network
    #     h = rect(linear(h) + linear(g_t))
    #
    #     #

    # ------------------------------------------------------------------------

    @application(inputs=['features'], outputs=['targets', 'r', 'c'])
    def classify(self, features):
        batch_size = features.shape[0]
        # Sample from mean-zeros std.-one Gaussian
        u = self.theano_rng.normal(
            size=(self.n_iter, batch_size, 1),
            avg=0., std=1.)

        # y, r, c, center_x, center_y, delta, sigma = self.apply(x=features, dummy=u)
        y, r, c = self.apply(x=features, dummy=u)

        return y, r, c
