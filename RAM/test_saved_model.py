import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset

with open('./log/RAM3D/potcup-simple-20161204-192319/potcup', "rb") as f:
    p = load(f, 'model')

if isinstance(p, Model):
    model = p

ram = model.get_top_bricks()[0]
dtensor5 = tensor.TensorType('float32', (False,) * 5)
x = dtensor5('input')  # keyword from fuel
y = tensor.matrix('targets')  # keyword from fuel
l, y_hat = ram.classify(x)
f = theano.function([x], [l, y_hat])

mnist_train = H5PYDataset('./data/potcup_vox.hdf5', which_sets=('train',))
handle = mnist_train.open()
model_idx = 2
train_data = mnist_train.get_data(handle, slice(model_idx , model_idx +1))
xx = train_data[0]
YY = train_data[1]
print(xx.shape)
l, prob = f(xx)
print(l)
print(YY)
print(prob)

import matplotlib.pyplot as plt
plt.figure()
for i in range(8):
    a = prob[:, 0, i]
    c = range(0,a.shape[0])
    plt.plot(c,a)
a = prob[:,0,YY[0,0]]
c = range(0, a.shape[0])
plt.plot(c, a, 'r')
plt.show()

import numpy as np
# for [y, r, c, cx, cy, cz] in f(train_features[0,:,:,:].reshape(1,32*32*32)):
#     print(cx, cy, cz)
cx,cy,cz = l[:,0,0],l[:,0,1],l[:,0,2]
'''visualize 3D data'''
def plot_cube(ax, x, y, z, inc, a):
    "x y z location and alpha"
    ax.plot_surface([[x, x + inc], [x, x + inc]], [[y, y], [y + inc, y + inc]], z, alpha=a,facecolors='y')
    ax.plot_surface([[x, x + inc], [x, x + inc]], [[y, y], [y + inc, y + inc]], z + inc, alpha=a,facecolors='y')

    ax.plot_surface(x, [[y, y], [y + inc, y + inc]], [[z, z + inc], [z, z + inc]], alpha=a,facecolors='y')
    ax.plot_surface(x + inc, [[y, y], [y + inc, y + inc]], [[z, z + inc], [z, z + inc]], alpha=a,facecolors='y')

    ax.plot_surface([[x, x], [x + inc, x + inc]], y, [[z, z + inc], [z, z + inc]], alpha=a,facecolors='y')
    ax.plot_surface([[x, x], [x + inc, x + inc]], y + inc, [[z, z + inc], [z, z + inc]], alpha=a,facecolors='y')

def viz2(V,cx,cy,cz):

    x = y = z = t = []
    x1 = y1 = z1 = t1 = []
    x2 = y2 = z2 = t2 = []
    x3 = y3 = z3 = t3 = []
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            for k in range(V.shape[2]):
                if V[i, j, k] != 0:
                    if (V[i, j, k] > 1e-1):
                        x = x + [i]
                        y = y + [j]
                        z = z + [k]
                        t = t + [V[i, j, k]]
                    if i==15:
                        y1 = y1 + [j]
                        z1 = z1 + [k]
                        t1 = t1 + [V[i, j, k]]
                    if j==15:
                        x2 = x2 + [i]
                        z2 = z2 + [k]
                        t2 = t2 + [V[i, j, k]]
                    if k==15:
                        x3 = x3 + [i]
                        y3 = y3 + [j]
                        t3 = t3 + [V[i, j, k]]

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    t = np.asarray(t)
    y1 = np.asarray(y1)
    z1 = np.asarray(z1)
    t1 = np.asarray(t1)
    x2 = np.asarray(x2)
    z2 = np.asarray(z2)
    t2 = np.asarray(t2)
    x3 = np.asarray(x3)
    y3 = np.asarray(y3)
    t3 = np.asarray(t3)

    # # slice along axis
    # fig, axes = plt.subplots(nrows=2, ncols=2,)
    #
    # ax1 = axes.flat[1]
    # im = ax1.scatter(y1, z1, c=t1, marker='o', s=30)
    # plt.xlim(0, V.shape[0])
    # plt.ylim(0, V.shape[1])
    #
    # ax2 = axes.flat[2]
    # im = ax2.scatter(x2, z2, c=t2, marker='o', s=30)
    # plt.xlim(0, V.shape[0])
    # plt.ylim(0, V.shape[1])
    #
    # ax3 = axes.flat[3]
    # im = ax3.scatter(x3, y3, c=t3, marker='o', s=30)
    # plt.xlim(0, V.shape[0])
    # plt.ylim(0, V.shape[1])
    #
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(x, y, z, c=t, marker='o', s=10, alpha=0.2)
    im = ax.scatter(cx, cy, cz, c=range(1,cx.shape[0]+1,1), marker='s', s=30)
    d = 5
    for i in range(len(cx)):
        plot_cube(ax, cx[i]-d/2, cy[i]-d/2, cz[i]-d/2, d, float(i)/len(cx))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.xlim(0, V.shape[0])
    plt.ylim(0, V.shape[1])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()
    plt.hold(True)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
viz2(train_data[0][0][0].reshape(32,32,32),cx,cy,cz)
print(cx)
print(cy)
print(cz)