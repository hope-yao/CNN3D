'''save data to hdf5 format, from Max'''

import numpy as np
import h5py
import tarfile, os
import sys
import cStringIO as StringIO
import tarfile
import time
import zlib

PREFIX = 'data/'
SUFFIX = '.npy.z'

class NpyTarReader(object):
    def __init__(self, fname):
        self.tfile = tarfile.open(fname, 'r|')

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        entry = self.tfile.next()
        if entry is None:
            raise StopIteration()
        name = entry.name[len(PREFIX):-len(SUFFIX)]
        fileobj = self.tfile.extractfile(entry)
        buf = zlib.decompress(fileobj.read())
        arr = np.load(StringIO.StringIO(buf))
        return arr, name

    def close(self):
        self.tfile.close()

train_dataset = NpyTarReader('c:\users\p2admin\documents/max/projects/cnn3d/more_data_mix/shapenet10_train.tar')
test_dataset = NpyTarReader('c:\users\p2admin\documents/max/projects/cnn3d/more_data_mix/shapenet10_test.tar')

train_features = []
train_targets = []
test_features = []
test_targets = []
for index, (array, name) in enumerate(train_dataset):
    if int(name[-3:])==1:
#         train_features.append(array.flatten())
        train_features.append(array.reshape(1,array.shape[0],array.shape[1],array.shape[2])*4.-1.)
        if int(name[0:3])==1:
            train_targets.append([1,1,1])
        else:
            train_targets.append([0,1,1])
for index, (array, name) in enumerate(test_dataset):
    if int(name[-3:]) == 1:
#         test_features.append(array.flatten())
        test_features.append(array.reshape(1,array.shape[0],array.shape[1],array.shape[2])*4.-1.)
        if int(name[0:3]) == 1:
            test_targets.append([1, 1, 1])
        else:
            test_targets.append([0, 1, 1])
        # test_targets.append([int(name[0:3])])

train_features = np.array(train_features)
train_targets = np.array(train_targets) #starts from 0
test_features = np.array(test_features)
test_targets = np.array(test_targets)
train_n, c, p1, p2, p3 = train_features.shape
test_n = test_features.shape[0]
n = train_n + test_n

f = h5py.File('potcup_org_function.hdf5', mode='w')
features = f.create_dataset('input', (n, c, p1, p2, p3), dtype='uint8')
targets = f.create_dataset('targets', (n, 3), dtype='uint8')

features[...] = np.vstack([train_features, test_features])
targets[...] = np.vstack([train_targets, test_targets])

features.dims[0].label = 'batch'
features.dims[1].label = 'input'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'index'

from fuel.datasets.hdf5 import H5PYDataset
split_dict = {
    'train': {'input': (0, train_n), 'targets': (0, train_n)},
    'test': {'input': (train_n, n), 'targets': (train_n, n)}}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.flush()
f.close()