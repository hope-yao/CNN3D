import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
import RAM_model as RAM_model

# with open('../log/RAM3D/read_small/potcup-simple-20161206-181231/potcup', "rb") as f:
with open('./bmnist-simple-20161208-132422/bmnist', "rb") as f:
        p = load(f, 'model')

if isinstance(p, Model):
    model = p

ram = model.get_top_bricks()[0]
x = tensor.tensor4('input')  # keyword from fuel
y = tensor.matrix('targets')  # keyword from fuel
l, prob, rho_orig, rho_larger, rho_largest = ram.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.
f = theano.function([x], [l, prob, rho_orig, rho_larger, rho_largest])

mnist_train = H5PYDataset('../data/bmnist.hdf5', which_sets=('test',))
handle = mnist_train.open()
model_idx = 100
train_data = mnist_train.get_data(handle, slice(model_idx , model_idx +1))
xx = train_data[0]
YY = train_data[1]
print(xx.shape)
l, prob, rho_orig, rho_larger, rho_largest = f(xx)
l = l*28
print(l)
print(YY)
print(prob)

import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(221)
for i in range(10):
    a = prob[:, 0, i]
    c = range(0,a.shape[0])
    plt.plot(c,a, label='%s data' % i)
plt.legend()

a = prob[:,0,YY[0,0]]
c = range(0, a.shape[0])
plt.scatter(c, a, s=40)

ax2 = fig.add_subplot(222)
plt.imshow(xx.reshape(28,28), cmap='Greys', interpolation='nearest', extent=[0,28,0,28])
ax2.set_xlim([0, 28])
ax2.set_ylim([0, 28])

plt.plot(l[:,0,1],l[:,0,0],'r.')
for i in range(ram.n_iter):
    y = l[i,0,0]
    x = l[i,0,1]
    ax2.text(x , y, i, fontsize=15, color='red')
    import matplotlib.patches as patches
    p = patches.Rectangle(
        (x-ram.read_N/2. , y-ram.read_N/2.), ram.read_N, ram.read_N,
        fill=False, clip_on=False, color='red'
        )
    ax2.add_patch(p)

# ax = fig.add_subplot(223)
# plt.imshow(rho_orig[0,0,:].reshape(ram.read_N,ram.read_N), cmap='Greys', interpolation='nearest')
#
# ax = fig.add_subplot(224)
# plt.imshow(rho_larger[0], cmap='Greys', interpolation='nearest')

plt.show(True)


