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
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.transformers import Flatten

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam, Momentum, Scale, AdaDelta, AdaGrad
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

from blocks_extras.extensions.plot import Plot

from RAM_model import *

sys.setrecursionlimit(100000)


# ----------------------------------------------------------------------------

def main(dataset, epochs, batch_size, learning_rate, attention,
         n_iter):


    # ---------------------------2D MNIST-------------------------------------

    if dataset=='mnist':
        image_size = (28, 28)
        channels = 1
        img_ndim = 2
        data_train = MNIST(which_sets=["train"], sources=['features', 'targets'])
        data_test = MNIST(which_sets=["test"], sources=['features', 'targets'])
        train_stream = DataStream.default_stream(data_train, iteration_scheme=SequentialScheme(data_train.num_examples, batch_size))
        # valid_stream = Flatten(
        #     DataStream.default_stream(data_valid, iteration_scheme=SequentialScheme(data_valid.num_examples, batch_size)))
        test_stream = DataStream.default_stream(data_test, iteration_scheme=SequentialScheme(data_test.num_examples, batch_size))
    elif dataset == 'potcup':
        image_size = (32,32,32)
        channels = 1
        img_ndim = 3
        from fuel.datasets.hdf5 import H5PYDataset
        train_set = H5PYDataset('../data/potcup_hollow_vox.hdf5', which_sets=('train',))
        train_stream = DataStream.default_stream(train_set,iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size))
        test_set = H5PYDataset('../data/potcup_hollow_vox.hdf5', which_sets=('test',))
        test_stream = DataStream.default_stream(test_set,iteration_scheme=ShuffledScheme(test_set.num_examples, batch_size))
    elif dataset == 'shapenet':
        image_size = (32,32,32)
        channels = 1
        img_ndim = 3
        from fuel.datasets.hdf5 import H5PYDataset
        train_set = H5PYDataset('../data/shapenet10.hdf5', which_sets=('train',))
        train_stream = DataStream.default_stream(train_set,iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size))
        test_set = H5PYDataset('../data/shapenet10.hdf5', which_sets=('test',))
        test_stream = DataStream.default_stream(test_set,iteration_scheme=ShuffledScheme(test_set.num_examples, batch_size))

    subdir = dataset + time.strftime("%Y%m%d-%H%M%S")
    # ---------------------------RAM_blocks SETUP-------------------------------------
    ram = RAM(image_size=image_size, channels=channels, attention=attention, n_iter=n_iter)
    ram.push_initialization_config()
    ram.initialize()

    # ---------------------------COMPILE-------------------------------------
    if img_ndim==2:
        x = tensor.ftensor4('features')  # keyword from fuel
    elif img_ndim == 3:
        dtensor5 = T.TensorType('float32', (False,) * 5)
        x = dtensor5('input')  # keyword from fuel
    y = tensor.matrix('targets')  # keyword from fuel
    l, y_hat, _, _, _ = ram.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.
    y_hat_last = y_hat[-1, :, :]  # pay attention to its shape and perhaps should use argmax?
    y_int = T.cast(y, 'int64')

    # ----------------------------COST----------------------------------
    cost = (CategoricalCrossEntropy().apply(y_int.flatten(), y_hat_last).copy(name='cost'))
    error = (MisclassificationRate().apply(y_int.flatten(), y_hat_last).copy(name='error'))
    cost.name ='cost'
    error.name ='error'

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
        # step_rule =AdaDelta()
        # step_rule = AdaGrad()
        step_rule=RMSProp(1e-3),
        # step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
        # step_rule=Scale(learning_rate=learning_rate)
    )

    # -------------------------Setup monitors--------------------------------------
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    # -------------------------MAIN LOOP--------------------------------------
    main_loop = MainLoop(
        model=Model(cost),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
                    Timing(),
                    FinishAfter(after_n_epochs=epochs),
                    TrainingDataMonitoring(
                       [cost, error],
                       prefix="train",
                       after_epoch=True),
                    DataStreamMonitoring(
                       [cost, error],
                       test_stream,
                       prefix="test"),
                    Checkpoint("{}/{}".format(subdir, dataset), save_main_loop=False, before_training=True,
                              after_epoch=True, save_separately=['log', 'model']),
                    ProgressBar(),
                    Printing(),
                    Plot(subdir, channels=[["train_cost","test_cost"], ["train_error","test_error"]], after_epoch=True)
                    ]
    )

    main_loop.run()

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, dest="dataset",
                        default="bmnist", help="Dataset to use: [mnist|bmnist|shapenet|potcup]")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=500, help="how many epochs")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                        default=64, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                        default=1e-2, help="Learning rate")
    parser.add_argument("--attention", "-a", type=int, default=5,
                        help="Use attention mechanism (read_window)")
    parser.add_argument("--n-iter", type=int, dest="n_iter",
                        default=5, help="number of time iteration in RNN")  # dim should be the number of classes
    args = parser.parse_args()
    main(**vars(args))