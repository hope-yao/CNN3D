#potcup2d20161217-192959 default data, ram, attention=5, iter=5, n0=16, n1=16, h=16 test error=1.0
#potcup2d20161217-194151 ux, ram, attention=5, iter=5, n0=16, n1=16, h=16 test error=0.4
#potcup2d20161217-202154 ux, ram, attention=5, iter=8, n0=16, n1=16, h=16 test error=1.0
#potcup2d20161217-224939 ux, conv_ram, attention=5, iter=5, test error=0.0

#potcup2d20161218-142737 default data, conv_ram, attention=28, iter=1, test_error=0.0
# Finding: the initial attention window is moved to the top left corner to only capture the nozzle! Does this mean the attention mechanism itself is working?
# Finding2: actually this is because the initial attention window by default is at (0,0)! when force move it to (14,14), get 0.6 test error...as shown in potcup2d20161218-153203
# Then try iter=2, got 0.2 test error, but the second window moves to the right bottom corner, doesn't make sense! see potcup2d20161218-153609
#








import theano
from theano import tensor
from blocks.serialization import load
from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
import numpy
import conv_RAM_model as RAM_model

# with open('../log/RAM/bmnist20161208-224205/bmnist', "rb") as f:
with open('./potcup2d20161218-212318/potcup2d', "rb") as f:
        p = load(f, 'model')

if isinstance(p, Model):
    model = p

ram = model.get_top_bricks()[0]
x = tensor.tensor4('input')  # keyword from fuel
y = tensor.matrix('targets')  # keyword from fuel
l, prob, h0, rho_orig = ram.classify(x)  # directly use theano to build the graph? Might be able to track iteration idx.
f = theano.function([x], [l, prob, h0, rho_orig])

mnist_train = H5PYDataset('../data/2dpotcup/potcup2d_point.hdf5', which_sets=('train',))
handle = mnist_train.open()
model_idx = 5
train_data = mnist_train.get_data(handle, slice(model_idx , model_idx +1))
xx = train_data[0]
YY = train_data[1]
print(xx.shape)
l, prob, h0, rho_orig = f(xx)
print(l)
print(YY)
print(prob)

################################################################################
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(5, 5)
# gs1.update(left=0.05, right=0.48, wspace=0.05)
ax_mnist = plt.subplot(gs1[0:3, 0:3])
ax_mnist.axis('equal')

ax_acc = plt.subplot(gs1[0:3,3:5])

ax_glimpse0 = plt.subplot(gs1[3,0])
ax_glimpse1 = plt.subplot(gs1[3,1])
ax_glimpse2 = plt.subplot(gs1[3,2])
ax_glimpse3 = plt.subplot(gs1[3,3])
ax_glimpse4 = plt.subplot(gs1[3,4])
ax_glimpse0.axis('equal')
ax_glimpse1.axis('equal')
ax_glimpse2.axis('equal')
ax_glimpse3.axis('equal')
ax_glimpse4.axis('equal')

ax_canvas0 = plt.subplot(gs1[4,0])
ax_canvas1 = plt.subplot(gs1[4,1])
ax_canvas2 = plt.subplot(gs1[4,2])
ax_canvas3 = plt.subplot(gs1[4,3])
ax_canvas4 = plt.subplot(gs1[4,4])
ax_canvas0.axis('equal')
ax_canvas1.axis('equal')
ax_canvas2.axis('equal')
ax_canvas3.axis('equal')
ax_canvas4.axis('equal')

ax_mnist.imshow(xx.reshape(28,28), cmap='Greys', interpolation='nearest')
# ax_mnist.set_xlim([0, 28])
# ax_mnist.set_ylim([0, 28])
for i in range(ram.n_iter):
    x = l[i,0,0]*28
    y = l[i,0,1]*28
    ax_mnist.text(x , y, i, fontsize=15, color='red')
    import matplotlib.patches as patches
    p = patches.Rectangle(
        (x-ram.read_N/2. , y-ram.read_N/2.), ram.read_N, ram.read_N,
        fill=False, clip_on=False, color='red'
        )
    ax_mnist.add_patch(p)

t = prob[:,0,:]
ax_acc.imshow(t.transpose(), interpolation='nearest', cmap=plt.cm.viridis,extent=[0,5,10,0])
# ax_acc.xlabel('time iteration')
# ax_acc.ylabel('class index')
# ax_acc.colorbar()

import numpy
glimpse_idx = 0
glimpse0 = numpy.zeros((28,28))
canvas0 = numpy.zeros((28,28))
x_start = 0
x_end = 28
y_start = 0
y_end = 28
# glimpse_idx = glimpse_idx + 1
# x_start = old_l[glimpse_idx,0,1]*28-ram.read_N/2.
# x_end = old_l[glimpse_idx,0,1]*28+ram.read_N/2.
# y_start = old_l[glimpse_idx,0,0]*28-ram.read_N/2.
# y_end = old_l[glimpse_idx,0,0]*28+ram.read_N/2.
glimpse0[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(28,28)
canvas0 = canvas0 + glimpse0
ax_glimpse0.imshow(glimpse0, cmap='Greys', interpolation='nearest')
ax_canvas0.imshow(canvas0, cmap='Greys', interpolation='nearest')
ax_glimpse0.get_xaxis().set_visible(False)
ax_glimpse0.get_yaxis().set_visible(False)
ax_canvas0.get_xaxis().set_visible(False)
ax_canvas0.get_yaxis().set_visible(False)

# glimpse_idx = 1
# glimpse1 = numpy.zeros((28,28))
# canvas1 = numpy.zeros((28,28))
# x_start = 0
# x_end = 28
# y_start = 0
# y_end = 28
# # x_start = old_l[glimpse_idx,0,1]*28-ram.read_N/2.
# # x_end = old_l[glimpse_idx,0,1]*28+ram.read_N/2.
# # y_start = old_l[glimpse_idx,0,0]*28-ram.read_N/2.
# # y_end = old_l[glimpse_idx,0,0]*28+ram.read_N/2.
# # glimpse_idx = glimpse_idx + 1
# glimpse1[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(28,28)
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
# x_start = 0
# x_end = 28
# y_start = 0
# y_end = 28
# # x_start = l[glimpse_idx,0,1]-ram.read_N/2.
# # x_end = l[glimpse_idx,0,1]+ram.read_N/2.
# # y_start = l[glimpse_idx,0,0]-ram.read_N/2.
# # y_end = l[glimpse_idx,0,0]+ram.read_N/2.
# # glimpse_idx = glimpse_idx + 1
# glimpse2[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(28,28)
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
# x_start = 0
# x_end = 28
# y_start = 0
# y_end = 28
# # x_start = l[glimpse_idx,0,1]-ram.read_N/2.
# # x_end = l[glimpse_idx,0,1]+ram.read_N/2.
# # y_start = l[glimpse_idx,0,0]-ram.read_N/2.
# # y_end = l[glimpse_idx,0,0]+ram.read_N/2.
# # glimpse_idx = glimpse_idx + 1
# glimpse3[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(28,28)
# canvas3 = canvas2 + glimpse3
# ax_glimpse3.imshow(glimpse3, cmap='Greys', interpolation='nearest')
# ax_canvas3.imshow(canvas3, cmap='Greys', interpolation='nearest')
# ax_glimpse3.get_xaxis().set_visible(False)
# ax_glimpse3.get_yaxis().set_visible(False)
# ax_canvas3.get_xaxis().set_visible(False)
# ax_canvas3.get_yaxis().set_visible(False)
#
#
# glimpse_idx = 4
# glimpse4 = numpy.zeros((28,28))
# canvas4 = numpy.zeros((28,28))
# x_start = 0
# x_end = 28
# y_start = 0
# y_end = 28
# # x_start = l[glimpse_idx,0,1]-ram.read_N/2.
# # x_end = l[glimpse_idx,0,1]+ram.read_N/2.
# # y_start = l[glimpse_idx,0,0]-ram.read_N/2.
# # y_end = l[glimpse_idx,0,0]+ram.read_N/2.
# # glimpse_idx = glimpse_idx + 1
# glimpse4[x_start:x_end,y_start:y_end] = rho_orig[glimpse_idx,0,:].reshape(28,28)
# canvas4 = canvas3 + glimpse4
# # ax_glimpse0.get_xaxis().set_visible(False)
# # ax_glimpse0.get_yaxis().set_visible(False)
# ax_glimpse4.imshow(glimpse4, cmap='Greys', interpolation='nearest')
# ax_canvas4.imshow(canvas4, cmap='Greys', interpolation='nearest')
# ax_glimpse4.get_xaxis().set_visible(False)
# ax_glimpse4.get_yaxis().set_visible(False)
# ax_canvas4.get_xaxis().set_visible(False)
# ax_canvas4.get_yaxis().set_visible(False)

plt.show(True)

# print(rho_orig)