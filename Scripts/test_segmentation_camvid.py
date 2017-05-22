import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
import glob

from pprint import pprint

from sklearn.preprocessing import normalize
caffe_root = '/home/bisipl/Documents/MATLAB/SegNet/caffe-segnet/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
# parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

pathes = glob.glob("/home/bisipl/Documents/MATLAB/SegNet/Skin/Validation/Val_data/*.png")
pathes.sort()
print(pathes)

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
# # transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
# # transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
# transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

for i in range(0, len(pathes)):

	# imageC = caffe.io.load_image(pathes[i])
	# plt.figure()
	# plt.imshow(imageC , vmin=0, vmax=1)
	# net.blobs['data'].data[...] = transformer.preprocess('data', imageC)

	# print(vars(net.blobs['data']))

	net.forward()

	image = net.blobs['data'].data
	label = net.blobs['label'].data
	predicted = net.blobs['prob'].data
	image = np.squeeze(image[0,:,:,:])
	output = np.squeeze(predicted[0,:,:,:])
	ind = np.argmax(output, axis=0)

	r = ind.copy()
	g = ind.copy()
	b = ind.copy()
	r_gt = label.copy()
	g_gt = label.copy()
	b_gt = label.copy()

	Skin = [0,0,0]
	Melanoma = [255,255,255]
	# Pole = [192,192,128]
	# Road_marking = [255,69,0]
	# Road = [128,64,128]
	# Pavement = [60,40,222]
	# Tree = [128,128,0]
	# SignSymbol = [192,128,128]
	# Fence = [64,64,128]
	# Car = [64,0,128]
	# Pedestrian = [64,64,0]
	# Bicyclist = [0,128,192]
	Unlabelled = [0,0,0]

	label_colours = np.array([Skin, Melanoma, Unlabelled])
	for l in range(0,3):
		r[ind==l] = label_colours[l,0]
		g[ind==l] = label_colours[l,1]
		b[ind==l] = label_colours[l,2]
		r_gt[label==l] = label_colours[l,0]
		g_gt[label==l] = label_colours[l,1]
		b_gt[label==l] = label_colours[l,2]

	rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb[:,:,0] = r/255.0
	rgb[:,:,1] = g/255.0
	rgb[:,:,2] = b/255.0
	rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb_gt[:,:,0] = r_gt/255.0
	rgb_gt[:,:,1] = g_gt/255.0
	rgb_gt[:,:,2] = b_gt/255.0

	image = image/255.0

	image = np.transpose(image, (1,2,0))
	output = np.transpose(output, (1,2,0))
	image = image[:,:,(2,1,0)]

	nameArray = pathes[i].split("/")
	name = nameArray[len(nameArray) - 1]
	print(name)
	scipy.misc.toimage(rgb, cmin=0.0, cmax=1).save('/home/bisipl/Documents/MATLAB/SegNet/Skin/Net_Label/' + name)

	# plt.figure()
	# plt.imshow(image,vmin=0, vmax=1)
	# # plt.figure()
	# # plt.imshow(rgb_gt,vmin=0, vmax=1)
	# plt.figure()
	# plt.imshow(rgb,vmin=0, vmax=1)
	# plt.show()




print 'Success!'

