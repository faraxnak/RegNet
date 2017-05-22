import caffe
import numpy as np
import matplotlib.pyplot as plt


class DiceLoss(caffe.Layer):
    """
    Compute energy based on dice coefficient.
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

        dice = np.zeros(bottom[0].data.shape[0],dtype=np.float32)
        self.union = np.zeros(bottom[0].data.shape[0],dtype=np.float32)
        self.intersection = np.zeros(bottom[0].data.shape[0],dtype=np.float32)

        self.result = np.squeeze(np.argmax(bottom[0].data[...],axis=1))
        self.gt = np.squeeze(bottom[1].data[...])
        # print(self.gt.shape, self.result.shape)


        self.gt = (self.gt > 0.5).astype(dtype=np.float32)
        self.result = self.result.astype(dtype=np.float32)

        for i in range(0,bottom[0].data.shape[0]):
            # compute dice
            CurrResult = (self.result[i,:]).astype(dtype=np.float32)
            CurrGT = (self.gt[i,:]).astype(dtype=np.float32)

            self.union[i]=(np.sum(CurrResult) + np.sum(CurrGT))
            self.intersection[i]=(np.sum(CurrResult * CurrGT))
            # print(CurrResult, CurrGT)
            dice[i] = 2 * self.intersection[i] / (self.union[i]+0.00001)
            # print ("for image ", i ," in batch dice coeff is ", dice[i])

        top[0].data[0]= 1 - np.sum(dice) / bottom[0].data.shape[0]
        # print("----------------------------")

    def backward(self, top, propagate_down, bottom):
            
            prob = bottom[0].data[...]
            bottom[0].diff[...] = np.zeros(bottom[0].diff.shape, dtype=np.float32)

            for i in range(0, bottom[0].diff.shape[0]):
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
                    bottom[0].diff[i, j, :] = 2 * sign * (self.gt[i, :] / self.union[i] 
                                                        - self.result[i, :]*(self.intersection[i])
                                                         / ((self.union[i]) ** 2))
                    #print(np.sum(bottom[0].diff[i, j , :]))
                # plt.figure()
                # plt.imshow(bottom[0].diff[i, 0, :],vmin=0, vmax=1)
                
            # plt.show()

                # bottom[0].diff[i, 1, :] -= 2.0 * ((self.gt[i, :] * self.union[i]) / ((self.union[i]) ** 2) - 2.0*prob[i,1,:]*(self.intersection[i]) / ((self.union[i]) ** 2))

    # def forward(self, bottom, top):
    #     self.diff[...] = bottom[0].data - bottom[1].data
    #     top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    # def backward(self, top, propagate_down, bottom):
    #     for i in range(2):
    #         if not propagate_down[i]:
    #             continue
    #         if i == 0:
    #             sign = 1
    #         else:
    #             sign = -1
    #         bottom[i].diff[...] = sign * self.diff / bottom[i].num
    #         # print(bottom[i].num, bottom[i].diff.shape)
    #     # for i in range(0, bottom[0].diff.shape[0]):
    #     #         print(np.sum(bottom[0].diff[i, 0 , :]))
    #     #         plt.figure()
    #     #         plt.imshow(bottom[0].diff[i, 0, :],vmin=0, vmax=1)
                
    #     # plt.show()