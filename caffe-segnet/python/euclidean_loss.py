import caffe
import numpy as np
import sys


class EuclideanLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """
    result = None
    index = None

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        self.result = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.index = np.zeros_like(bottom[1].data, dtype=np.uint16)
        for i in range(0,bottom[0].data.shape[0]):
            estimate = np.squeeze(bottom[0].data[i,:])
            ind = np.argmax(estimate, axis=0)
            self.index[i] = ind
            for j in range(0,estimate.shape[0]):
                self.result[i, 0, ind == j] = j
        # if bottom[0].count != self.result.count:
        #     raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        diff = self.result - bottom[1].data
        print(self.index.shape)
        sys.stdin.read(1)
        self.diff[self.index] = diff
        top[0].data[...] = np.sum(diff**2) / bottom[0].num / 2.
        # loss = np.zeros(bottom[0].data.shape[0], dtype=np.float32)
        # for i in range(0,bottom[0].data.shape[0]):
        #     self.diff[i,:] = bottom[0].data - bottom[1].data
        #     loss[i] = np.sum(self.diff[i,:]**2) / bottom[i,0].num / 2.
        #     # top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.
        # top[0].data[0] = np.sum(loss)

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num