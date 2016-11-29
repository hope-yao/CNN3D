import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset

with open('bmnist-simple-20161129-132952/bmnist', "rb") as f:
    p = load(f, 'model')

if isinstance(p, Model):
    model = p

ram = model.get_top_bricks()[0]
x = tensor.ftensor4('features')  # keyword from fuel
y = tensor.matrix('targets')  # keyword from fuel
l, y_hat, h0, h1, h2,_ = ram.classify(x)
f = theano.function([x], [l, y_hat])

mnist_train = H5PYDataset('../data/mnist.hdf5', which_sets=('train',))
handle = mnist_train.open()
train_data = mnist_train.get_data(handle, slice(0, 4))
xx = train_data[0]
YY = train_data[1]
print(xx.shape)
l, prob = f(xx)
print(l)
print(YY)
print(prob)