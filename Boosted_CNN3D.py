

try:
    from blocks_extras.extensions.plot import Plot
    BLOCKS_EXTRAS_AVAILABLE = True
except:
    BLOCKS_EXTRAS_AVAILABLE = False



import logging
from argparse import ArgumentParser
import theano
import theano.tensor as T
import numpy

theano.config.floatX = 'float32'
floatX = theano.config.floatX

def setting(save_to):
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
    parser.add_argument("--batch-size", type=int, default=18,
                        help="Batch size.")
    parser.add_argument("--datafile-hdf5", default='C:/Users/p2admin/documents/max/projects/CNN3D/potcup_vox.hdf5', nargs="?",
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
    return numpy.float32(new_weight)

if __name__ == "__main__":
    # '''Adaboost'''
    # from sklearn.cross_validation import cross_val_score
    # from sklearn.datasets import load_iris
    # from sklearn.ensemble import AdaBoostClassifier
    #
    # iris = load_iris()
    # clf = AdaBoostClassifier(n_estimators=15)
    # scores = cross_val_score(clf, iris.data, iris.target)
    # scores.mean()

    import imp
    from path import Path
    cfg_dir = Path("C:/Users/p2admin/documents/max/projects/CNN3D/LeNet3D.py")
    LeNet3D = imp.load_source("train_cnn3d", cfg_dir)

    # ensemble N LeNet together
    N = 2

    n = 18
    weight = numpy.ones(n)/float(n) # weight on incorrect training sample
    for i in range(N):
        save_to = "LeNet3D_"+str(i)+".pkl"
        args = setting(save_to)
        LeNet3D.train_cnn3d(weight, **vars(args))
        predv, realv = LeNet3D.forward_cnn3d(**vars(args))
        weight = boosting(predv,realv,weight)
        print(weight)
    # i = 0
    # save_to = "LeNet3D_"+str(i)+".pkl"
    # args = setting(save_to)
    # convnet = LeNet3D.test(weight, **vars(args))
    # dtensor5 = T.TensorType(floatX, (False,) * 5)
    # x = dtensor5(name='input')
    # t = convnet.apply(x)
    # do_classify = theano.function([x], outputs=t)
    # y = do_classify(numpy.random.normal(size=(18, 1, 32, 32, 32)).astype('float32'))
    # a = 1


    # y = tensor.lmatrix('targets')
    # probs = convnet.apply(x)
    # true_dist = y.flatten()
    # coding_dist = probs
    # entropy = theano.tensor.nnet.categorical_crossentropy(coding_dist, true_dist)





