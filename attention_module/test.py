import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset

with open('/home/hope-yao/Documents/CNN3D/attention_module/bmnist-simple-20161122-110759/bmnist', "rb") as f:
    p = load(f, 'model')

if isinstance(p, Model):
    model = p

ram = model.get_top_bricks()[0]
x = tensor.ftensor4('features')  # keyword from fuel
y = tensor.matrix('targets')  # keyword from fuel
l, prob = ram.classify(x)
f = theano.function([x], [l, prob])

mnist_train = H5PYDataset('../data/mnist.hdf5', which_sets=('test',))
handle = mnist_train.open()
train_data = mnist_train.get_data(handle, slice(0, 4))
xx = train_data[0]
YY = train_data[1]
print(xx.shape)
l, prob = f(xx)
print(l)
print(YY)
print(prob)