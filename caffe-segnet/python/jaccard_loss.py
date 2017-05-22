import caffe
import numpy as np
import matplotlib.pyplot as plt


class JaccardLoss(caffe.Layer):
    """
    Compute energy based on jaccard coefficient.
    """
    union = None
    intersection = None
    result = None
    gt = None

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute the dice. the result of the softmax and the ground truth.")



    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != 2*bottom[1].count:
            print bottom[0].data.shape
            print bottom[1].data.shape
            raise Exception("the dimension of inputs should match")

        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

        # loss output is two scalars (mean and std)
        top[0].reshape(1)

    def forward(self, bottom, top):

        jaccard = np.zeros(bottom[0].data.shape[0],dtype=np.float32)
        self.union = np.zeros(bottom[0].data.shape[0],dtype=np.float32)
        self.intersection = np.zeros(bottom[0].data.shape[0],dtype=np.float32)

        self.result = np.squeeze(np.argmax(bottom[0].data[...],axis=1))
        self.gt = np.squeeze(bottom[1].data[...])
        # print(self.gt.shape, self.result.shape)


        self.gt = (self.gt > 0.5).astype(dtype=np.float32)
        self.result = self.result.astype(dtype=np.float32)

        for i in range(0,bottom[0].data.shape[0]):
            # compute jaccard
            CurrResult = (self.result[i,:]).astype(dtype=np.float32)
            CurrGT = (self.gt[i,:]).astype(dtype=np.float32)
            # print(np.amax(CurrGT))

            self.union[i]=(np.sum(CurrResult) + np.sum(CurrGT))
            self.intersection[i]=(np.sum(CurrResult * CurrGT))
            # print(CurrResult, CurrGT)
            jaccard[i] = self.intersection[i] / (self.union[i]+0.00001 - self.intersection[i])
            # print ("for image ", i ," in batch jaccard coeff is ", jaccard[i])

        top[0].data[0]= 1 - np.sum(jaccard) / bottom[0].data.shape[0]
        # print("----------------------------")

    def backward(self, top, propagate_down, bottom):
            
        prob = bottom[0].data[...]
        bottom[0].diff[...] = np.zeros(bottom[0].diff.shape, dtype=np.float32)

        for i in range(0, bottom[0].diff.shape[0]):
            # print(np.amax(self.result[i, :]))
            # plt.figure()
            # plt.imshow(self.result[i, :], vmin= 0, vmax = 1)
            # plt.show()
            # print(self.union[i], ((self.union[i]) ** 2), np.sum(self.gt[i, :] * self.union[i]))
            for j in range(2):

                if not propagate_down[j]:
                    continue
                if j == 0:
                    sign = 1
                else:
                    sign = -1
                # bottom[0].diff[i, j, :] = sign * (2.0 * (self.gt[i, :] * self.union[i] / self.union[i] 
                                                    # - 2.0*prob[i,1,:]*(self.intersection[i]) / ((self.union[i]) ** 2)))
                n_pixels = bottom[0].diff[i,j,:].shape[0] * bottom[0].diff[i,j,:].shape[1] #np.sqrt(n_pixels) * 
                bottom[0].diff[i, j, :] = sign * (self.gt[i, :] / (self.union[i] -  self.intersection[i])
                                                    - (2 * self.result[i, :] - self.gt[i, :]) * (self.intersection[i])
                                                     / ((self.union[i] -  self.intersection[i]) ** 2))