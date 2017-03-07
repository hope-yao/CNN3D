import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
from LeNet_Mnist import LeNet

# with open('./bmnist20161220-114356/bmnist', "rb") as f:
with open('./LeNet', "rb") as f:
    p = load(f, 'model')

if isinstance(p, Model):
    model = p

lenet = model.get_top_bricks()[0]

x = tensor.tensor4('features')
y = tensor.lmatrix('targets')

# Normalize input and apply the convnet
probs = lenet.apply(x)
f = theano.function([x], [probs])

mnist_train = H5PYDataset('../data/bmnist.hdf5', which_sets=('test',))
handle = mnist_train.open()
model_idx = 4
train_data = mnist_train.get_data(handle, slice(model_idx , model_idx +1))
xx = train_data[0]
YY = train_data[1]
tt = f(xx)
print(tt)

# activation of the first convolutional layer before pooling
act1 = lenet.layers[1].apply(lenet.layers[0].apply(x))
ff = theano.function([x],[act1])
aa = ff(xx)

# plot it
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
ax_act = plt.subplot(1,1,1)
ax_act.axis('equal')
ax_act.imshow(aa[0][0][0], cmap='Greys', interpolation='nearest')
# ax_act.get_xaxis().set_visible(False)
# ax_act.get_yaxis().set_visible(False)
plt.show(True)
