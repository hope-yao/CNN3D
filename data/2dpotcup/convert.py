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
from scipy.io import loadmat

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
PREFIX = 'c:/users/p2admin/documents/max/projects/cnn3d/data/2dpotcup'
SUFFIX = '.png'

train_features = []
train_targets = []
test_features = []
test_targets = []

for i in range(1,6):
    # img = rgb2gray(mpimg.imread(PREFIX + '/pot' + str(i) + '.png'))
    # train_features.append(1-img.reshape(1,28,28))
    img = loadmat('c:/users/p2admin/documents/max/projects/cnn3d/data/2dpotcup/pot' + str(i) + '2.mat').get('u')
    train_features.append(img.reshape(1,28,28))
    train_targets.append([0])
for i in range(1,6):
    # img = rgb2gray(mpimg.imread(PREFIX + '/cup' + str(i) + '.png'))
    # train_features.append(1-img.reshape(1,28,28))
    img = loadmat('c:/users/p2admin/documents/max/projects/cnn3d/data/2dpotcup/cup' + str(i) + '2.mat').get('u')
    train_features.append(img.reshape(1,28,28))
    train_targets.append([1])
for i in range(1,6):
    # img = rgb2gray(mpimg.imread(PREFIX + '/cup' + str(i) + '_test.png'))
    # test_features.append(1-img.reshape(1,28,28))
    img = loadmat('c:/users/p2admin/documents/max/projects/cnn3d/data/2dpotcup/cup' + str(i) + '_test2.mat').get('u')
    test_features.append(img.reshape(1,28,28))
    test_targets.append([1])

train_features = np.array(train_features)
train_targets = np.array(train_targets) #starts from 0
test_features = np.array(test_features)
test_targets = np.array(test_targets)
train_n, c, p1, p2 = train_features.shape
test_n = test_features.shape[0]
n = train_n + test_n

f = h5py.File(PREFIX +'/potcup2d_point.hdf5', mode='w')
features = f.create_dataset('features', (n, c, p1, p2), dtype='float32')
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



#################################################################################################
####### try flow
# __author__ = 'yiren'
# import os.path
# import pickle
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# # for image preprocessing
# import lic_internal
#
# # get normalized gradient from sobel filter
# def sobel(img):
#     gradx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
#     grady = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
#     grad = np.dstack((gradx, grady))
#     grad_norm = np.linalg.norm(grad, axis=2)
#     st = np.dstack((gradx*gradx, grady*grady, gradx*grady))
#     return grad, grad_norm, st
#
# # main function to update tangent
# # ita, r: wm parameter, neighbourhood radius
# def edge_tangent_flow(img):
#     # step 1: Calculate the struture tensor
#     grad, grad_norm, st = sobel(img)
#     row_, col_ = img.shape
#
#     # step 2: Gaussian blur the struct tensor. sst_sigma = 2.0
#     sigma_sst = 2.0
#     gaussian_size = int((sigma_sst*2)*2+1)
#     blur = cv2.GaussianBlur(st, (gaussian_size,gaussian_size), sigma_sst)
#
#     tan_ETF = np.zeros((row_,col_,2))
#     E = blur[:,:,0]
#     G = blur[:,:,1]
#     F = blur[:,:,2]
#
#     lambda2 = 0.5*(E+G-np.sqrt((G-E)*(G-E)+4.0*F*F))
#     v2x = (lambda2 - G != 0) * (lambda2 - G) + (lambda2 - G == 0) * F
#     v2y = (lambda2 - G != 0) * F + (lambda2 - G == 0) * (lambda2 -E)
#     # v2x = cv2.GaussianBlur(v2x, (gaussian_size,gaussian_size), sigma_sst)
#     # v2y = cv2.GaussianBlur(v2y, (gaussian_size,gaussian_size), sigma_sst)
#     v2 = np.sqrt(v2x*v2x+v2y*v2y)
#     tan_ETF[:,:,0] = v2x/(v2+0.0000001)*((v2!=0)+0)
#     tan_ETF[:,:,1] = v2y/(v2+0.0000001)*((v2!=0)+0)
#
#     # plt.subplot(1,3,1)
#     # plt.imshow(tan_ETF[:,:,0],'gray')
#     # plt.subplot(1,3,2)
#     # plt.imshow(tan_ETF[:,:,1],'gray')
#     return tan_ETF
#
# # Visualize a vector field by using LIC (Linear Integral Convolution).
# def visualizeByLIC(vf):
#     row_,col_,dep_ = vf.shape
#     texture = np.random.rand(col_,row_).astype(np.float32)
#     kernellen=9
#     kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
#     kernel = kernel.astype(np.float32)
#     vf = vf.astype(np.float32)
#     img = lic_internal.line_integral_convolution(vf, texture, kernel)
#     return img
#
# for i in range(1,6):
#     img = rgb2gray(mpimg.imread(PREFIX + '/pot' + str(i) + '.png'))
#     tf = np.transpose(edge_tangent_flow(img), (2,0,1))
#     # vtf = visualizeByLIC(tf)
#     # plt.imshow(vtf, 'gray')
#     train_features.append(tf.reshape(2,28,28))
#     train_targets.append([0])
# for i in range(1,6):
#     img = rgb2gray(mpimg.imread(PREFIX + '/cup' + str(i) + '.png'))
#     tf = np.transpose(edge_tangent_flow(img), (2, 0, 1))
#     train_features.append(tf.reshape(2,28,28))
#     train_targets.append([1])
# for i in range(1,6):
#     img = rgb2gray(mpimg.imread(PREFIX + '/cup' + str(i) + '_test.png'))
#     tf = np.transpose(edge_tangent_flow(img), (2, 0, 1))
#     test_features.append(tf.reshape(2,28,28))
#     test_targets.append([0])
#
# train_features = np.array(train_features)
# train_targets = np.array(train_targets) #starts from 0
# test_features = np.array(test_features)
# test_targets = np.array(test_targets)
# train_n, c, p1, p2 = train_features.shape
# test_n = test_features.shape[0]
# n = train_n + test_n
#
# f = h5py.File(PREFIX +'/potcup2d_flow.hdf5', mode='w')
# features = f.create_dataset('features', (n, c, p1, p2), dtype='uint8')
# targets = f.create_dataset('targets', (n, 1), dtype='uint8')
#
# features[...] = np.vstack([train_features, test_features])
# targets[...] = np.vstack([train_targets, test_targets])
#
# features.dims[0].label = 'batch'
# features.dims[1].label = 'input'
# targets.dims[0].label = 'batch'
# targets.dims[1].label = 'index'
#
# from fuel.datasets.hdf5 import H5PYDataset
# split_dict = {
#     'train': {'features': (0, train_n), 'targets': (0, train_n)},
#     'test': {'features': (train_n, n), 'targets': (train_n, n)}}
# f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
#
# f.flush()
# f.close()