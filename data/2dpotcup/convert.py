'''save data to hdf5 format, from Max'''

import numpy as np
import h5py
import tarfile, os
import sys
import cStringIO as StringIO
import tarfile
import time
import zlib
import matplotlib.image as mpimg
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
PREFIX = 'c:/users/p2admin/documents/max/projects/cnn3d/data/2dpotcup'
SUFFIX = '.png'

train_features = []
train_targets = []
test_features = []
test_targets = []

for i in range(1,6):
    img = rgb2gray(mpimg.imread(PREFIX + '/pot' + str(i) + '.png'))
    train_features.append(img.reshape(1,28,28))
    train_targets.append([0])
for i in range(1,6):
    img = rgb2gray(mpimg.imread(PREFIX + '/cup' + str(i) + '.png'))
    train_features.append(img.reshape(1,28,28))
    train_targets.append([1])
for i in range(1,6):
    img = rgb2gray(mpimg.imread(PREFIX + '/cup' + str(i) + '_test.png'))
    test_features.append(img.reshape(1,28,28))
    test_targets.append([0])

train_features = np.array(train_features)
train_targets = np.array(train_targets) #starts from 0
test_features = np.array(test_features)
test_targets = np.array(test_targets)
train_n, c, p1, p2 = train_features.shape
test_n = test_features.shape[0]
n = train_n + test_n

f = h5py.File(PREFIX +'/potcup2d.hdf5', mode='w')
features = f.create_dataset('features', (n, c, p1, p2), dtype='uint8')
targets = f.create_dataset('targets', (n, 1), dtype='uint8')

features[...] = np.vstack([train_features, test_features])
targets[...] = np.vstack([train_targets, test_targets])

features.dims[0].label = 'batch'
features.dims[1].label = 'input'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'index'

from fuel.datasets.hdf5 import H5PYDataset
split_dict = {
    'train': {'features': (0, train_n), 'targets': (0, train_n)},
    'test': {'features': (train_n, n), 'targets': (train_n, n)}}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.flush()
f.close()