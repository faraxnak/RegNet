import caffe
import numpy as np
import matplotlib.pyplot as plt
import imutils
import cv2


class TestRotationTranslation(caffe.Layer):
    # ref = none
    # im = none
    # label = none

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need four inputs. ref, float, estimated and label")

    def reshape(self, bottom, top):
      print(bottom[0].count)
      print(bottom[1].count)
      print(bottom[2].count)
        # check input dimensions match
        # if bottom[0].count != bottom[1].count:
        #     raise Exception("Inputs must have the same dimension.")
        # if bottom[1].count != bottom[2].count:
        #     raise Exception("Inputs must have the same dimension.")

    def forward(self, bottom, top):
      for i in range(0,bottom[0].data.shape[0]):
            ref = np.squeeze(bottom[0].data[i,:])
            fl = np.squeeze(bottom[1].data[i,:])
            estimate = bottom[2].data[i,:]
            # ind = np.argmax(estimate, axis=0)
            # print(ind)
            # estimateImg = ind.copy()
            # for j in range(0,estimate.shape[0]):
            #   estimateImg[ind == j] = j
            # estimateImg = np.squeeze(estimate)
            # print(estimate.shape)
            label = bottom[3].data[i,:]
            print('rotation in degrees:', label[0])
            print('estimate in degrees: ', estimate[0])
            print('translaion in pixels:', label[1], label[2])
            print('translaion estimate in pixels:', estimate[1], estimate[2])
            rotated = imutils.rotate(fl, -1 * estimate[0])
            M = np.float32([[1, 0, -1*estimate[1]],[0, 1, -1*estimate[2]]])
            rows,cols = rotated.shape
            translated = cv2.warpAffine(rotated,M,(cols,rows))
            # print(estimateImg.max(axis=1))
            plt.figure()
            plt.imshow(ref, cmap='gray')
            plt.figure()
            plt.imshow(fl, cmap='gray')
            plt.figure()
            plt.imshow(translated, cmap='gray')
            plt.show()

    def backward(self, top, propagate_down, bottom):
        pass
        