#!/usr/bin/env python


from cnn3d_bricks import Convolutional3, MaxPooling3, ConvolutionalSequence3, Flattener3



try:
    from blocks_extras.extensions.plot import Plot
    BLOCKS_EXTRAS_AVAILABLE = True
except:
    BLOCKS_EXTRAS_AVAILABLE = False



import logging
import numpy
from argparse import ArgumentParser
import theano
from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,
                           Softmax, Activation)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.datasets.hdf5 import H5PYDataset
from toolz.itertoolz import interleave
import theano.tensor as T
from blocks.initialization import Constant, Uniform

theano.config.floatX = 'float32'
floatX = theano.config.floatX


class LeNet(FeedforwardSequence, Initializable):
    """LeNet-like convolutional network.

    The class implements LeNet, which is a convolutional sequence with
    an MLP on top (several fully-connected layers). For details see
    [LeCun95]_.

    .. [LeCun95] LeCun, Yann, et al.
       *Comparison of learning algorithms for handwritten digit
       recognition.*,
       International conference on artificial neural networks. Vol. 60.

    Parameters
    ----------
    conv_activations : list of :class:`.Brick`
        Activations for convolutional network.
    num_channels : int
        Number of channels in the input image.
    image_shape : tuple
        Input image shape.
    filter_sizes : list of tuples
        Filter sizes of :class:`.blocks.conv.ConvolutionalLayer`.
    feature_maps : list
        Number of filters for each of convolutions.
    pooling_sizes : list of tuples
        Sizes of max pooling for each convolutional layer.
    top_mlp_activations : list of :class:`.blocks.bricks.Activation`
        List of activations for the top MLP.
    top_mlp_dims : list
        Numbers of hidden units and the output dimension of the top MLP.
    conv_step : tuples
        Step of convolution (similar for all layers).
    border_mode : str
        Border mode of convolution (similar for all layers).

    """
    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes,
                 top_mlp_activations, top_mlp_dims,
                 conv_step=None, border_mode='valid', **kwargs):
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
                                                   image_size=image_shape)

        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.flattener = Flattener3()

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)

        application_methods = [self.conv_sequence.apply, self.flattener.apply,
                               self.top_mlp.apply]
        super(LeNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [numpy.prod(conv_out_dim)] + self.top_mlp_dims


def run_cnn3d(save_to, num_epochs, feature_maps=None, mlp_hiddens=None,
         conv_sizes=None, pool_sizes=None, batch_size=100,
         num_batches=None):
    if feature_maps is None:
        feature_maps = [16, 32]
    if mlp_hiddens is None:
        mlp_hiddens = [200]
    if conv_sizes is None:
        conv_sizes = [5, 5, 5]
    if pool_sizes is None:
        pool_sizes = [2, 2, 2]
    image_size = (32, 32, 32)
    output_size = 10

    # Use ReLUs everywhere and softmax for the final prediction
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]
    convnet = LeNet(conv_activations, 1, image_size,
                    filter_sizes=[(5,5,5),(5,5,5)],
                    feature_maps=feature_maps,
                    pooling_sizes=[(2,2,2),(2,2,2)],
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [output_size],
                    border_mode='valid',
                    weights_init=Uniform(width=.2),
                    biases_init=Constant(0))
    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    convnet.push_initialization_config()
    convnet.layers[0].weights_init = Uniform(width=.2)
    convnet.layers[1].weights_init = Uniform(width=.09)
    convnet.top_mlp.linear_transformations[0].weights_init = Uniform(width=.08)
    convnet.top_mlp.linear_transformations[1].weights_init = Uniform(width=.11)
    convnet.initialize()
    logging.info("Input dim: {} {} {} {}".format(
        *convnet.children[0].get_dim('input_')))
    for i, layer in enumerate(convnet.layers):
        if isinstance(layer, Activation):
            logging.info("Layer {} ({})".format(
                i, layer.__class__.__name__))
        else:
            logging.info("Layer {} ({}) dim: {} {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))

    dtensor5 = T.TensorType(floatX, (False,) * 5)
    x = dtensor5(name='input')
    y = tensor.lmatrix('targets')

    # Normalize input and apply the convnet
    probs = convnet.apply(x)
    cost = (CategoricalCrossEntropy().apply(y.flatten(), probs)
            .copy(name='cost'))
    error_rate = (MisclassificationRate().apply(y.flatten(), probs)
                  .copy(name='error_rate'))

    cg = ComputationGraph([cost, error_rate])
    from blocks.filter import VariableFilter
    from blocks.roles import PARAMETER

    params = VariableFilter(roles=[PARAMETER])(cg.variables)

    train_set = H5PYDataset('shapenet10.hdf5', which_sets=('train',))
    train_set_stream = DataStream.default_stream(
        train_set, iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size))
    test_set = H5PYDataset('shapenet10.hdf5', which_sets=('test',))
    test_set_stream = DataStream.default_stream(
        test_set, iteration_scheme=ShuffledScheme(
            test_set.num_examples, batch_size))

    # Train with simple SGD
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=Scale(learning_rate=0.1))
    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs,
                              after_n_batches=num_batches),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      test_set_stream,
                      prefix="test"),
                  TrainingDataMonitoring(
                      [cost, error_rate,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_to),
                  ProgressBar(),
                  Printing()]

    model = Model(cost)

    main_loop = MainLoop(
        algorithm,
        train_set_stream,
        model=model,
        extensions=extensions)

    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a convolutional network "
                            "on 3D ShapeNet10 dataset.")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="LeNet3D.pkl", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    parser.add_argument("--feature-maps", type=int, nargs='+',
                        default=[16, 24], help="List of feature maps numbers.")
    parser.add_argument("--mlp-hiddens", type=int, nargs='+', default=[250],
                        help="List of numbers of hidden units for the MLP.")
    parser.add_argument("--conv-sizes", type=int, nargs='+', default=[5, 5, 5],
                        help="Convolutional kernels sizes. The kernels are "
                        "always square.")
    parser.add_argument("--pool-sizes", type=int, nargs='+', default=[2, 2, 2],
                        help="Pooling sizes. The pooling windows are always "
                             "square. Should be the same length as "
                             "--conv-sizes.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size.")
    args = parser.parse_args()
    run_cnn3d(**vars(args))
