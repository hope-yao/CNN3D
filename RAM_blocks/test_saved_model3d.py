import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
from viz import plot_cube
from viz import viz3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('./potcup20161209-142816/potcup', "rb") as f:
# with open('./shapenet-simple-20161208-140611/shapenet', "rb") as f:
        p = load(f, 'model')

if isinstance(p, Model):
    model = p

ram = model.get_top_bricks()[0]
dtensor5 = tensor.TensorType('float32', (False,) * 5)
x = dtensor5('input')  # keyword from fuel
y = tensor.matrix('targets')  # keyword from fuel
l, y_hat, rho_orig, rho_larger, rho_largest = ram.classify(x)
f = theano.function([x], [l, y_hat, rho_orig, rho_larger, rho_largest])

mnist_train = H5PYDataset('../data/potcup_hollow_vox.hdf5', which_sets=('train',))
# mnist_train = H5PYDataset('../data/shapenet10.hdf5', which_sets=('train',))
handle = mnist_train.open()
model_idx = 1
train_data = mnist_train.get_data(handle, slice(model_idx , model_idx +1))
xx = train_data[0]
YY = train_data[1]
print(xx.shape)
ll, prob, rho_orig, rho_larger, rho_largest = f(xx)
print(ll)
print(YY)
print(prob)

########change according to number of classes
l = ll*32
cx,cy,cz = ll[:,0,0]*32,ll[:,0,1]*32,ll[:,0,2]*32
# plot_cube(ax,cx,cy,cz,ram.read_N)
print(cx)
print(cy)
print(cz)

V = train_data[0][0][0].reshape(32,32,32)
x,y,z,t = viz3(V,0)


################################################################################
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(5, 5)
# gs1.update(left=0.05, right=0.48, wspace=0.05)
ax_mnist = plt.subplot(gs1[0:3, 0:3], projection='3d')
ax_mnist.axis('equal')

ax_acc = plt.subplot(gs1[0:3,3:5])

ax_glimpse0 = plt.subplot(gs1[3,0], projection='3d')
ax_glimpse1 = plt.subplot(gs1[3,1], projection='3d')
ax_glimpse2 = plt.subplot(gs1[3,2], projection='3d')
ax_glimpse3 = plt.subplot(gs1[3,3], projection='3d')
ax_glimpse4 = plt.subplot(gs1[3,4], projection='3d')
ax_glimpse0.axis('equal')
ax_glimpse1.axis('equal')
ax_glimpse2.axis('equal')
ax_glimpse3.axis('equal')
ax_glimpse4.axis('equal')

ax_canvas0 = plt.subplot(gs1[4,0], projection='3d')
ax_canvas1 = plt.subplot(gs1[4,1], projection='3d')
ax_canvas2 = plt.subplot(gs1[4,2], projection='3d')
ax_canvas3 = plt.subplot(gs1[4,3], projection='3d')
ax_canvas4 = plt.subplot(gs1[4,4], projection='3d')
ax_canvas0.axis('equal')
ax_canvas1.axis('equal')
ax_canvas2.axis('equal')
ax_canvas3.axis('equal')
ax_canvas4.axis('equal')

ax_mnist.scatter(x, y, z, c=t, marker='o', s=10)
ax_mnist.set_xlabel('X Label')
ax_mnist.set_ylabel('Y Label')
ax_mnist.set_zlabel('Z Label')
plt.xlim(0, V.shape[0])
plt.ylim(0, V.shape[1])
ax_mnist.set_zlim(0, V.shape[2])

for i in range(ram.n_iter-1):
    x = cx[0]
    y = cy[1]
    z = cz[2]
    plot_cube(ax_mnist,x-ram.read_N/2. , y-ram.read_N/2.,z-ram.read_N/2., ram.read_N/2., 0.1, i)

t = prob[:,0,:]
ax_acc.imshow(t.transpose(), interpolation='nearest', cmap=plt.cm.viridis,extent=[0,t.shape[0],t.shape[1],0])
# # ax_acc.xlabel('time iteration')# # ax_acc.xlabel('time iteration')
# # ax_acc.ylabel('class index')
# # ax_acc.colorbar()

import numpy
glimpse_idx = 0
glimpse0 = numpy.zeros((32,32,32))
canvas0 = numpy.zeros((32,32,32))
x_start = l[glimpse_idx,0,0]-ram.read_N/2.
x_end = l[glimpse_idx,0,0]+ram.read_N/2.
y_start = l[glimpse_idx,0,1]-ram.read_N/2.
y_end = l[glimpse_idx,0,1]+ram.read_N/2.
z_start = l[glimpse_idx,0,2]-ram.read_N/2.
z_end = l[glimpse_idx,0,2]+ram.read_N/2.
glimpse_idx = glimpse_idx + 1
glimpse0[x_start:x_end,y_start:y_end,z_start:z_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N,ram.read_N)
canvas0 = canvas0 + glimpse0
ax_glimpse0.imshow(glimpse0, cmap='Greys', interpolation='nearest')
ax_canvas0.imshow(canvas0, cmap='Greys', interpolation='nearest')
ax_glimpse0.get_xaxis().set_visible(False)
ax_glimpse0.get_yaxis().set_visible(False)
ax_canvas0.get_xaxis().set_visible(False)
ax_canvas0.get_yaxis().set_visible(False)
#
# glimpse_idx = 1
# glimpse1 = numpy.zeros((28,28))
# canvas1 = numpy.zeros((28,28))
# x_start = l[glimpse_idx,0,1]-ram.read_N/2.
# x_end = l[glimpse_idx,0,1]+ram.read_N/2.
# y_start = l[glimpse_idx,0,0]-ram.read_N/2.
# y_end = l[glimpse_idx,0,0]+ram.read_N/2.
# glimpse_idx = glimpse_idx + 1
# glimpse1[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N)
# canvas1 = canvas0 + glimpse1
# # ax_glimpse0.get_xaxis().set_visible(False)
# # ax_glimpse0.get_yaxis().set_visible(False)
# ax_glimpse1.imshow(glimpse1, cmap='Greys', interpolation='nearest')
# ax_canvas1.imshow(canvas1, cmap='Greys', interpolation='nearest')
# ax_glimpse1.get_xaxis().set_visible(False)
# ax_glimpse1.get_yaxis().set_visible(False)
# ax_canvas1.get_xaxis().set_visible(False)
# ax_canvas1.get_yaxis().set_visible(False)
#
# glimpse_idx = 2
# glimpse2 = numpy.zeros((28,28))
# canvas2 = numpy.zeros((28,28))
# x_start = l[glimpse_idx,0,1]-ram.read_N/2.
# x_end = l[glimpse_idx,0,1]+ram.read_N/2.
# y_start = l[glimpse_idx,0,0]-ram.read_N/2.
# y_end = l[glimpse_idx,0,0]+ram.read_N/2.
# glimpse_idx = glimpse_idx + 1
# glimpse2[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N)
# canvas2 = canvas1 + glimpse2
# ax_glimpse2.imshow(glimpse2, cmap='Greys', interpolation='nearest')
# ax_canvas2.imshow(canvas2, cmap='Greys', interpolation='nearest')
# ax_glimpse2.get_xaxis().set_visible(False)
# ax_glimpse2.get_yaxis().set_visible(False)
# ax_canvas2.get_xaxis().set_visible(False)
# ax_canvas2.get_yaxis().set_visible(False)
#
# glimpse_idx = 3
# glimpse3 = numpy.zeros((28,28))
# canvas3 = numpy.zeros((28,28))
# x_start = l[glimpse_idx,0,1]-ram.read_N/2.
# x_end = l[glimpse_idx,0,1]+ram.read_N/2.
# y_start = l[glimpse_idx,0,0]-ram.read_N/2.
# y_end = l[glimpse_idx,0,0]+ram.read_N/2.
# glimpse_idx = glimpse_idx + 1
# glimpse3[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(ram.read_N,ram.read_N)
# canvas3 = canvas2 + glimpse3
# ax_glimpse3.imshow(glimpse3, cmap='Greys', interpolation='nearest')
# ax_canvas3.imshow(canvas3, cmap='Greys', interpolation='nearest')
# ax_glimpse3.get_xaxis().set_visible(False)
# ax_glimpse3.get_yaxis().set_visible(False)
# ax_canvas3.get_xaxis().set_visible(False)
# ax_canvas3.get_yaxis().set_visible(False)
