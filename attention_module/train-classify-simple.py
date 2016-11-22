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

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam, Momentum, Scale, AdaDelta
from blocks.bricks import Tanh, Identity
from blocks.bricks.cost import BinaryCrossEntropy, CategoricalCrossEntropy, MisclassificationRate
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model


from fuel.datasets.mnist import MNIST


try:
    from blocks.extras import Plot
except ImportError:
    pass

from draw_classify_simple import *

sys.setrecursionlimit(100000)


# ----------------------------------------------------------------------------

def main(dataset, epochs, batch_size, learning_rate, attention,
         n_iter):


    # ---------------------------2D MNIST-------------------------------------
    image_size = (28, 28)
    channels = 1

    data_train = MNIST(which_sets=["train"], sources=['features', 'targets'])
    data_test = MNIST(which_sets=["test"], sources=['features', 'targets'])
    train_stream = DataStream.default_stream(data_train, iteration_scheme=SequentialScheme(data_train.num_examples, batch_size))
    # valid_stream = Flatten(
    #     DataStream.default_stream(data_valid, iteration_scheme=SequentialScheme(data_valid.num_examples, batch_size)))
    test_stream = DataStream.default_stream(data_test, iteration_scheme=SequentialScheme(data_test.num_examples, batch_size))
    subdir = dataset + "-simple-" + time.strftime("%Y%m%d-%H%M%S")

    # ---------------------------RAM SETUP-------------------------------------
    ram = RAM(image_size=image_size, channels=channels, attention=attention, n_iter=n_iter)
    ram.push_initialization_config()
    ram.initialize()

    # ---------------------------COMPILE-------------------------------------
    x = tensor.ftensor4('features')  # keyword from fuel
    y = tensor.matrix('targets')  # keyword from fuel
    l, y_hat = ram.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.
    y_hat_last = y_hat[-1, :, :]  # pay attention to its shape and perhaps should use argmax?
    y_int = T.cast(y, 'int64')

    # ----------------------------COST----------------------------------
    cost = (CategoricalCrossEntropy().apply(y_int.flatten(), y_hat_last).copy(name='recognition'))
    error = (MisclassificationRate().apply(y_int.flatten(), y_hat_last).copy(name='error'))
    cg = ComputationGraph([cost])
    params = VariableFilter(roles=[PARAMETER])(cg.variables)

    # --------------------------ALGORITHM----------------------------------
    algorithm = GradientDescent(
        cost=cost,
        parameters=params,
        # step_rule=CompositeRule([
        #     StepClipping(10.),
        #     Adam(learning_rate),
        # ])
        step_rule =AdaDelta()
        # step_rule=RMSProp(learning_rate),
        # step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
        # step_rule=Scale(learning_rate=learning_rate)
    )
    # algorithm = AdaDelta()

    # -------------------------Setup monitors--------------------------------------
    monitors = [cost, error]

    train_monitors = monitors[:]
    # train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
    # train_monitors += [aggregation.mean(algorithm.total_step_norm)]
    # Live plotting...
    plot_channels = [
        ["train_total_gradient_norm", "train_total_step_norm"]
    ]
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    plotting_extensions = []
    # if live_plotting:
    #     plotting_extensions = [
    #         Plot(name, channels=plot_channels)
    #     ]

    # -------------------------MAIN LOOP--------------------------------------
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

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, dest="dataset",
                        default="bmnist", help="Dataset to use: [bmnist|mnist_lenet|cifar10]")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=200, help="how many epochs")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                        default=100, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                        default=5e-2, help="Learning rate")
    parser.add_argument("--attention", "-a", type=int, default=5,
                        help="Use attention mechanism (read_window)")
    parser.add_argument("--n-iter", type=int, dest="n_iter",
                        default=4, help="number of time iteration in RNN")  # dim should be the number of classes
    args = parser.parse_args()
    main(**vars(args))