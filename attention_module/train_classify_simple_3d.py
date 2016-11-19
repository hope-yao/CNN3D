# Hope: simple RNN model w/o VAE in 3D



#!/usr/bin/env python

from __future__ import division, print_function

import logging

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import os
import theano
import time

from argparse import ArgumentParser
from blocks.graph import ComputationGraph

from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

from blocks.algorithms import Scale
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.serialization import load
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream

theano.config.floatX = 'float32'
floatX = theano.config.floatX

import theano.tensor as tensor

try:
    from blocks.extras import Plot
except ImportError:
    pass

from draw_classify_simple_3d import *

sys.setrecursionlimit(100000)


# ----------------------------------------------------------------------------

def main(name, dataset, epochs, batch_size, learning_rate, attention,
         n_iter, rnn_dim, y_dim, oldmodel, live_plotting):

    # image_size, channels, data_train, data_valid, data_test = datasets.get_data(dataset)
    image_size = (32,32,32)
    channels = 1
    train_set = H5PYDataset('../data/shapenet10.hdf5', which_sets=('train',))
    test_set = H5PYDataset('../data/shapenet10.hdf5', which_sets=('test',))

    train_stream = Flatten(
        DataStream.default_stream(train_set, iteration_scheme=SequentialScheme(train_set.num_examples, batch_size)))
    test_stream = Flatten(
        DataStream.default_stream(test_set, iteration_scheme=SequentialScheme(test_set.num_examples, batch_size)))


    # train_stream = DataStream.default_stream(
    #     train_set, iteration_scheme=ShuffledScheme(
    #         train_set.num_examples, batch_size))
    # test_stream = DataStream.default_stream(
    #     test_set, iteration_scheme=ShuffledScheme(
    #         test_set.num_examples, batch_size))


    if name is None:
        name = dataset

    attention_tag = 'full'

    # Learning rate
    def lr_tag(value):
        """ Convert a float into a short tag-usable string representation. E.g.:
            0.1   -> 11
            0.01  -> 12
            0.001 -> 13
            0.005 -> 53
        """
        exp = np.floor(np.log10(value))
        leading = ("%e" % value)[0]
        return "%s%d" % (leading, -exp)

    lr_str = lr_tag(learning_rate)

    subdir = name + "-simple-" + time.strftime("%Y%m%d-%H%M%S")
    longname = "%s-%s-t%d-rnn%d-y%d-lr%s" % (dataset, attention_tag, n_iter, rnn_dim, y_dim, lr_str)
    pickle_file = subdir + "/" + longname + ".pkl"

    print("\nRunning experiment %s" % longname)
    print("               dataset: %s" % dataset)
    print("          subdirectory: %s" % subdir)
    print("         learning rate: %g" % learning_rate)
    print("             attention: %s" % attention)
    print("          n_iterations: %d" % n_iter)
    print("         rnn dimension: %d" % rnn_dim)
    print("           y dimension: %d" % y_dim)
    print("            batch size: %d" % batch_size)
    print("                epochs: %d" % epochs)
    print()

    # ----------------------------------------------------------------------
    from draw_classify_simple_3d import DrawClassifyModel3d

    draw = DrawClassifyModel3d(image_size=image_size, channels=channels, attention=attention)
    draw.push_initialization_config()
    draw.conv_sequence.layers[0].weights_init = Uniform(width=.2)
    draw.conv_sequence.layers[1].weights_init = Uniform(width=.09)
    draw.top_mlp.linear_transformations[0].weights_init = Uniform(width=.08)
    draw.top_mlp.linear_transformations[1].weights_init = Uniform(width=.11)
    draw.initialize()

    # ------------------------------------------------------------------------
    x = tensor.matrix('input') # keyword from fuel
    y = tensor.matrix('targets') # keyword from fuel
    # dtensor5 = T.TensorType(floatX, (False,) * 5)
    # x = dtensor5(name='input')
    # y = tensor.lmatrix('targets')

    y_hat, _, _ = draw.classify(x)

    y_hat_last = y_hat[-1,:,:] # output should be batch_size * class
    # y_hat_last = y_hat
    # # classification_error = -T.mean(T.log(y_hat_last)*y.astype(np.int64))
    y_int = tensor.cast(y, 'int64') #MAX: might go from 1-10? need to check
    # recognition = -T.mean(T.log(y_hat_last)[T.arange(batch_size), y_int]) # guess (rnn_iter (16), class (10), batch_size)
    # recognition.name = "recognition"
    #
    tol = 1e-4
    recognition_convergence = (-y_hat*tensor.log2(y_hat+tol)).sum(axis=2).sum(axis=0).mean()
    recognition_convergence.name = "recognition_convergence"


    # from LeNet
    recognition = (CategoricalCrossEntropy().apply(y_int.flatten(), y_hat_last)
            .copy(name='recognition'))
    error = (MisclassificationRate().apply(y_int.flatten(), y_hat_last)
                  .copy(name='error_rate'))

    cost = recognition
    # cost = recognition + recognition_convergence.mean()
    cost.name = "cost"

    # _, activated_id = T.max_and_argmax(y_hat_last, axis=1)
    # error = theano.tensor.neq(activated_id.flatten(), y_int.flatten()).sum()/float(batch_size)
    error.name = "error"
    from blocks.filter import VariableFilter
    from blocks.roles import PARAMETER
    from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam, Momentum
    from blocks.monitoring import aggregation
    from blocks.main_loop import MainLoop
    from blocks.model import Model
    from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
    from blocks.extensions.saveload import Checkpoint
    from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
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
    from blocks.algorithms import AdaDelta
    # algorithm = AdaDelta(
    #     cost=cost,
    #     parameters=params
    # )

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
    if live_plotting:
        plotting_extensions = [
            Plot(name, channels=plot_channels)
        ]

    if oldmodel is not None:
        print("Initializing parameters with old model %s" % oldmodel)
        with open(oldmodel, "rb") as f:
            # oldmodel = pickle.load(f)
            oldmodel = load(f, 'model')
            # main_loop.model.set_parameter_values(oldmodel.get_top_bricks()[0])
            main_loop = MainLoop(
                model=oldmodel,
                data_stream=train_stream,
                algorithm=AdaDelta,
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
                               Checkpoint("{}/{}".format(subdir, name), save_main_loop=False, before_training=True,
                                          after_epoch=True, save_separately=['log', 'model']),
                               # SampleCheckpoint(image_size=image_size[0], channels=channels, save_subdir=subdir,
                               #                  before_training=True, after_epoch=True),
                               ProgressBar(),
                               Printing()] + plotting_extensions)


        # del oldmodel
    else:
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
                           Checkpoint("{}/{}".format(subdir, name), save_main_loop=False, before_training=True,
                                      after_epoch=True, save_separately=['log', 'model']),
                           # SampleCheckpoint(image_size=image_size[0], channels=channels, save_subdir=subdir,
                           #                  before_training=True, after_epoch=True),
                           ProgressBar(),
                           Printing()] + plotting_extensions)



    main_loop.run()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--live-plotting", "--plot", action="store_true",
                        default=False, help="Activate live-plotting to a bokeh-server")
    parser.add_argument("--name", type=str, dest="name",
                        default=None, help="Name for this experiment")
    parser.add_argument("--dataset", type=str, dest="dataset",
                        default="bmnist", help="Dataset to use: [bmnist|mnist_lenet|cifar10]")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=100, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                        default=100, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                        default=1e-2, help="Learning rate")
    parser.add_argument("--attention", "-a", type=str,
                        default=32, help="Use attention mechanism (read_window)")
    parser.add_argument("--niter", type=int, dest="n_iter",
                        default=1, help="No. of iterations")
    parser.add_argument("--rnn-dim", type=int, dest="rnn_dim",
                        default=256, help="Encoder RNN state dimension") # originally 256
    parser.add_argument("--y-dim", type=int, dest="y_dim",
                        default=10, help="Decoder  RNN state dimension") # dim should be the number of classes
    parser.add_argument("--oldmodel", type=str,
                        help="Use a model pkl file created by a previous run as a starting point for all parameters")
    args = parser.parse_args()

    main(**vars(args))
