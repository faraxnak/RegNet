import caffe
import numpy as np
import sys
import matplotlib.pyplot as plt


class WeightedEuclideanLossLayer(caffe.Layer):
    """
    input is taken from softmax, output is calculated to be given to Euclidean Layer
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute weighted Euclidean loss.")

    def reshape(self, bottom, top):
        # check input dimensions match
        # wil check later!
        self.result = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.diff = np.zeros_like(bottom[0].data, dtype = np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        for i in range(0, bottom[0].data.shape[0]):
            prob = np.squeeze(bottom[0].data[i, :])
            for j in range(0,prob.shape[0]):
                self.result[i, 0, :] += prob[j, :] * j
        #     tmp = np.squeeze(self.result[i, 0])
        #     plt.figure()
        #     plt.imshow(tmp, vmin = 0, vmax = prob.shape[0])
        # plt.show()
        diff = self.result - bottom[1].data
        top[0].data[...] = np.sum(diff**2) / bottom[0].num / 2.

        for i in range(0, bottom[0].data.shape[0]):
            for j in range(0,prob.shape[0]):
                self.diff[i, j, :] = diff[i, 0, :] * (j - self.result[i, 0, :])

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num