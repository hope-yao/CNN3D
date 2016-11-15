

try:
    from blocks_extras.extensions.plot import Plot
    BLOCKS_EXTRAS_AVAILABLE = True
except:
    BLOCKS_EXTRAS_AVAILABLE = False



import logging
from argparse import ArgumentParser
import theano

theano.config.floatX = 'float32'
floatX = theano.config.floatX

if __name__ == "__main__":
    import imp
    from path import Path
    cfg_dir = Path("/home/hope-yao/Documents/CNN3D/LeNet3D.py")
    config_module = imp.load_source("run_cnn3d", cfg_dir)
    



    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a convolutional network "
                            "on 3D ShapeNet10 dataset.")
    parser.add_argument("--num-epochs", type=int, default=20,
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

    config_module.run_cnn3d(**vars(args))
