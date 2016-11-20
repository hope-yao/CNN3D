

try:
    from blocks_extras.extensions.plot import Plot
    BLOCKS_EXTRAS_AVAILABLE = True
except:
    BLOCKS_EXTRAS_AVAILABLE = False


import logging
from argparse import ArgumentParser
import theano
import numpy
from fuel.datasets.hdf5 import H5PYDataset

theano.config.floatX = 'float32'
floatX = theano.config.floatX

def setting(save_to,datafile_hdf5):
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a convolutional network "
                            "on 3D ShapeNet10 dataset.")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default=save_to, nargs="?",
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
    parser.add_argument("--datafile-hdf5", default=datafile_hdf5, nargs="?",
                        help="Training and testing data")
    args = parser.parse_args()
    return args

def boosting(predv,realv,weight):
    pred = numpy.argmax(predv,axis=1)
    epsilon = numpy.sum(numpy.abs((pred - realv.T))) / numpy.float32(predv.shape[0])
    alpha = 0.5 * numpy.log((1-epsilon)/epsilon)
    new_weight = weight
    for i,wi in enumerate(weight):
        if pred[i]==realv[i]:
            new_weight[i] = wi * numpy.exp(-alpha)
        else:
            new_weight[i] = wi * numpy.exp(alpha)
    new_weight = new_weight / numpy.sum(weight)
    return alpha, numpy.float32(new_weight)

if __name__ == "__main__":
    from bricks3D import cnn3d_bricks

    import imp
    from path import Path
    cfg_dir = Path("./boosting/LeNet3D.py")
    LeNet3D = imp.load_source("train_cnn3d", cfg_dir)

    # ensemble N LeNets together
    N = 10
    datafile_hdf5 = './data/potcup_vox.hdf5'
    train_set = H5PYDataset(datafile_hdf5, which_sets=('train',))
    n = train_set.num_examples# number of training data
    tt_alpha = numpy.zeros(N)
    predv = []
    predv_test = []
    tt_predv = []
    tt_predv_test = []
    realv = []
    realv_test = []
    weight = numpy.ones(n)/n # weight on incorrect training sample
    for i in range(N):
        save_to = "./pkl/LeNet3D_"+str(i)+".pkl"
        args = setting(save_to,datafile_hdf5)
        # LeNet3D.train_cnn3d(weight, **vars(args))  # training network use weighted error metric
        predv, realv, predv_test, realv_test = LeNet3D.forward_cnn3d(**vars(args)) # get network predictions
        err = numpy.argmax(predv, axis=1) - realv.T
        print(err)
        alpha, weight = boosting(predv,realv,weight)
        tt_predv = tt_predv + [predv]
        tt_predv_test = tt_predv_test+ [predv_test]
        tt_alpha[i] = alpha
        print('boosting idx:', i)

        # ensemble
        final_pred = numpy.zeros_like(predv)
        final_pred_test = numpy.zeros_like(predv_test)
        for ii in range(i+1):
            final_pred = final_pred + tt_predv[ii] * tt_alpha[ii]
            final_pred_test = final_pred_test + tt_predv_test[ii] * tt_alpha[ii]
        final_pred = final_pred / numpy.sum(tt_alpha)
        final_pred_test = final_pred_test / numpy.sum(tt_alpha)
        pred = numpy.argmax(final_pred , axis=1)
        epsilon = numpy.sum(numpy.abs((pred - realv.T))) / numpy.float32(realv.shape[0])
        pred_test = numpy.argmax(final_pred_test , axis=1)
        epsilon_test = numpy.sum(numpy.abs((pred_test - realv_test.T))) / numpy.float32(realv_test.shape[0])
        print(epsilon, epsilon_test)






