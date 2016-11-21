# MAX: simple RNN model w/o VAE



#!/usr/bin/env python

from __future__ import division, print_function

import logging
import numpy as np

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import os
import theano
import theano.tensor as T
import fuel
import ipdb
import time
import cPickle as pickle

from argparse import ArgumentParser
from theano import tensor

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam, Momentum, Scale
from blocks.bricks import Tanh, Identity
from blocks.bricks.cost import BinaryCrossEntropy, CategoricalCrossEntropy, MisclassificationRate
from blocks.bricks.conv import Convolutional, ConvolutionalSequence
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.serialization import load
# from leNet import *
# from fuel.datasets.binarized_mnist import BinarizedMNIST
from fuel.datasets.mnist import MNIST


try:
    from blocks.extras import Plot
except ImportError:
    pass

# import draw.datasets as datasets
from draw_classify_simple import *
from samplecheckpoint import SampleCheckpoint

sys.setrecursionlimit(100000)


# ----------------------------------------------------------------------------

def main(dataset, epochs, batch_size, learning_rate, attention,
         y_dim):

    image_size = (28, 28)
    channels = 1

    # image_size, channels, data_train, data_valid, data_test = datasets.get_data(dataset)
    data_train = MNIST(which_sets=["train"], sources=['features', 'targets'])
    data_test = MNIST(which_sets=["test"], sources=['features', 'targets'])

    train_stream = Flatten(
        DataStream.default_stream(data_train, iteration_scheme=SequentialScheme(data_train.num_examples, batch_size)))
    # valid_stream = Flatten(
    #     DataStream.default_stream(data_valid, iteration_scheme=SequentialScheme(data_valid.num_examples, batch_size)))
    test_stream = Flatten(
        DataStream.default_stream(data_test, iteration_scheme=SequentialScheme(data_test.num_examples, batch_size)))

    subdir = dataset + "-simple-" + time.strftime("%Y%m%d-%H%M%S")


    # ----------------------------------------------------------------------

    draw = DrawClassifyModel(image_size=image_size, channels=channels, attention=attention)
    draw.push_initialization_config()
    draw.initialize()
    # ------------------------------------------------------------------------
    x = tensor.matrix('features')  # keyword from fuel
    y = tensor.matrix('targets')  # keyword from fuel
    l, y_hat = ram.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.

    y_hat_last = prob[-1, :, :]  # pay attention to its shape and perhaps should use argmax?
    y_int = T.cast(y, 'int64')

    tol = 1e-4
    recognition_convergence = (-y_hat * T.log2(y_hat + tol)).sum(axis=2).mean(axis=1).flatten()
    recognition_convergence.name = "recognition_convergence"

    # from LeNet
    cost = (CategoricalCrossEntropy().apply(y_int.flatten(), y_hat_last)
                   .copy(name='recognition'))
    error = (MisclassificationRate().apply(y_int.flatten(), y_hat_last).copy(name='error'))

    # ------------------------------------------------------------
    cg = ComputationGraph([cost])
    params = VariableFilter(roles=[PARAMETER])(cg.variables)

    algorithm = GradientDescent(
        cost=cost,
        parameters=params,
        # step_rule=CompositeRule([
        #     StepClipping(10.),
        #     Adam(learning_rate),
        # ])
        # step_rule=RMSProp(learning_rate),
        # step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
        step_rule=Scale(learning_rate=learning_rate)
    )

    # ------------------------------------------------------------------------
    # Setup monitors
    monitors = [cost, error]

    train_monitors = monitors[:]
    train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
    train_monitors += [aggregation.mean(algorithm.total_step_norm)]
    # Live plotting...
    plot_channels = [
        ["train_total_gradient_norm", "train_total_step_norm"]
    ]

    # ------------------------------------------------------------

    if not os.path.exists(subdir):
        os.makedirs(subdir)

    plotting_extensions = []
    # if live_plotting:
    #     plotting_extensions = [
    #         Plot(name, channels=plot_channels)
    #     ]

    main_loop = MainLoop(
        model=Model(cost),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
                       Timing(),
                       FinishAfter(after_n_epochs=epochs),
                       TrainingDataMonitoring(
                           train_monitors,
                           prefix="train",
                           after_epoch=True),
                       #            DataStreamMonitoring(
                       #                monitors,
                       #                valid_stream,
                       ##                updates=scan_updates,
                       #                prefix="valid"),
                       DataStreamMonitoring(
                           monitors,
                           test_stream,
                           #                updates=scan_updates,
                           prefix="test"),
                       # Checkpoint(name, before_training=False, after_epoch=True, save_separately=['log', 'model']),
                       Checkpoint("{}/{}".format(subdir, dataset), save_main_loop=False, before_training=True,
                                  after_epoch=True, save_separately=['log', 'model']),
                       # SampleCheckpoint(image_size=image_size[0], channels=channels, save_subdir=subdir,
                       #                  before_training=True, after_epoch=True),
                       ProgressBar(),
                       Printing()] + plotting_extensions)

    main_loop.run()

    #
    # for iteration in range(1):
    #     oldmodel_address = 'C:\Users\p2admin\Documents\Max\Projects\draw/mnist_lenet-simple-20161101-232031/' #window 5, iter 3
    #     try:
    #         with open(oldmodel_address, "rb") as f:
    #             # oldmodel = pickle.load(f)
    #             oldmodel = load(f, 'model')
    #             draw = oldmodel.get_top_bricks()[0]
    #             draw.n_iter = n_iter
    #             # del oldmodel
    #             f.close()
    #     except:
    #         ## initialize trained model
    #         draw = DrawClassifyModel(image_size=image_size, channels=channels, attention=attention)
    #         draw.push_initialization_config()
    #         draw.conv_sequence.layers[0].weights_init = Uniform(width=.2)
    #         draw.conv_sequence.layers[1].weights_init = Uniform(width=.09)
    #         draw.top_mlp.linear_transformations[0].weights_init = Uniform(width=.08)
    #         draw.top_mlp.linear_transformations[1].weights_init = Uniform(width=.11)
    #         draw.initialize()
    #
    #         ## load existing model
    #         # trained_address = 'c:\users\p2admin\documents\max\projects\draw\mnist_lenet-simple-20161030-204133\mnist_lenet' # attention = 7, n_iter = 4
    #         # with open(trained_address, "rb") as f:
    #             # trained_model = load(f, 'model')
    #             # draw = trained_model.get_top_bricks()[0]
    #             # draw.conv_sequence = trained.conv_sequence
    #             # draw.top_mlp = trained.top_mlp
    #             # draw.n_iter = n_iter
    #         # f.close()
    #
    #
    #     # ------------------------------------------------------------------------
    #     x = tensor.matrix('features') # keyword from fuel
    #     y = tensor.matrix('targets') # keyword from fuel
    #
    #     y_hat, _, _, _, _ = draw.classify(x)
    #
    #     y_hat_last = y_hat[-1,:,:] # output should be batch_size * class
    #     # y_hat_last = y_hat
    #     # # classification_error = -T.mean(T.log(y_hat_last)*y.astype(np.int64))
    #     y_int = T.cast(y, 'int64')
    #     # recognition = -T.mean(T.log(y_hat_last)[T.arange(batch_size), y_int]) # guess (rnn_iter (16), class (10), batch_size)
    #     # recognition.name = "recognition"
    #     #
    #     tol = 1e-4
    #     recognition_convergence = (-y_hat*T.log2(y_hat+tol)).sum(axis=2).mean(axis=1).flatten()
    #     recognition_convergence.name = "recognition_convergence"
    #
    #
    #     # from LeNet
    #     recognition = (CategoricalCrossEntropy().apply(y_int.flatten(), y_hat_last)
    #             .copy(name='recognition'))
    #     # recognition_all = T.as_tensor_variable([CategoricalCrossEntropy().apply(y_int.flatten(), y_hat[i,:,:]).copy(name='recognition-%s' % i) for i in range(n_iter)]).copy(name='recognition_all')
    #     recognition_all = T.as_tensor_variable([-T.mean(T.log(y_hat[i,:,:])[T.arange(batch_size), y_int]) for i in range(n_iter)]).copy(name='recognition_all')
    #     error = (MisclassificationRate().apply(y_int.flatten(), y_hat_last).copy(name='error'))
    #
    #     if np.mod(iteration, 2)==0:
    #         cost = recognition
    #         # cost = recognition - recognition_convergence.mean()
    #     else:
    #         # cost = recognition + recognition_convergence.mean()
    #         # cost = T.mean(-theano.shared(2)*recognition_all * recognition_convergence + theano.shared(6)*recognition_all + recognition_convergence)
    #         cost = T.dot(-T.constant(2.0) * recognition_all * recognition_convergence + T.constant(-2.*np.log2(0.1)) * recognition_all
    #                      + T.constant(-np.log(0.1)) * recognition_convergence, T.constant(np.arange(n_iter)))
    #
    #     cost.name = "cost"
    #
    #     # _, activated_id = T.max_and_argmax(y_hat_last, axis=1)
    #     # error = theano.tensor.neq(activated_id.flatten(), y_int.flatten()).sum()/float(batch_size)
    #
    #     # ------------------------------------------------------------
    #     cg = ComputationGraph([cost])
    #     params = VariableFilter(roles=[PARAMETER])(cg.variables)
    #
    #     algorithm = GradientDescent(
    #         cost=cost,
    #         parameters=params,
    #         # step_rule=CompositeRule([
    #         #     StepClipping(10.),
    #         #     Adam(learning_rate),
    #         # ])
    #         # step_rule=RMSProp(learning_rate),
    #         # step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
    #         step_rule=Scale(learning_rate=learning_rate)
    #     )
    #
    #     # ------------------------------------------------------------------------
    #     # Setup monitors
    #     monitors = [cost, error]
    #
    #     train_monitors = monitors[:]
    #     train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
    #     train_monitors += [aggregation.mean(algorithm.total_step_norm)]
    #     # Live plotting...
    #     plot_channels = [
    #          ["train_total_gradient_norm", "train_total_step_norm"]
    #     ]
    #
    #     # ------------------------------------------------------------
    #
    #     if not os.path.exists(subdir):
    #         os.makedirs(subdir)
    #
    #     plotting_extensions = []
    #     # if live_plotting:
    #     #     plotting_extensions = [
    #     #         Plot(name, channels=plot_channels)
    #     #     ]
    #
    #     main_loop = MainLoop(
    #         model=Model(cost),
    #         data_stream=train_stream,
    #         algorithm=algorithm,
    #         extensions=[
    #                        Timing(),
    #                        FinishAfter(after_n_epochs=epochs),
    #                        TrainingDataMonitoring(
    #                            train_monitors,
    #                            prefix="train",
    #                            after_epoch=True),
    #                        #            DataStreamMonitoring(
    #                        #                monitors,
    #                        #                valid_stream,
    #                        ##                updates=scan_updates,
    #                        #                prefix="valid"),
    #                        DataStreamMonitoring(
    #                            monitors,
    #                            test_stream,
    #                            #                updates=scan_updates,
    #                            prefix="test"),
    #                        # Checkpoint(name, before_training=False, after_epoch=True, save_separately=['log', 'model']),
    #                        Checkpoint("{}/{}".format(subdir, dataset), save_main_loop=False, before_training=True,
    #                                   after_epoch=True, save_separately=['log', 'model']),
    #                        # SampleCheckpoint(image_size=image_size[0], channels=channels, save_subdir=subdir,
    #                        #                  before_training=True, after_epoch=True),
    #                        ProgressBar(),
    #                        Printing()] + plotting_extensions)
    #
    #     main_loop.run()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, dest="dataset",
                        default="bmnist", help="Dataset to use: [bmnist|mnist_lenet|cifar10]")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=5, help="how many epochs")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                        default=100, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                        default=1e-3, help="Learning rate")
    parser.add_argument("--attention", "-a", type=int, default=5,
                        help="Use attention mechanism (read_window)")
    parser.add_argument("--y-dim", type=int, dest="y_dim",
                        default=10, help="Decoder  RNN state dimension")  # dim should be the number of classes
    args = parser.parse_args()
    main(**vars(args))