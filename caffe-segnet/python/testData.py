import caffe
import numpy as np
import matplotlib.pyplot as plt


class TestData(caffe.Layer):
    # ref = none
    # im = none
    # label = none

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need three inputs.")

    def reshape(self, bottom, top):
    	# print(bottom[0].count)
    	# print(bottom[1].count)
    	# print(bottom[2].count)
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # if bottom[1].count != bottom[2].count:
        #     raise Exception("Inputs must have the same dimension.")

    def forward(self, bottom, top):
    	for i in range(0,bottom[0].data.shape[0]):
        	ref = np.squeeze(bottom[0].data[i,:])
        	im = np.squeeze(bottom[1].data[i,:])
        	label = np.squeeze(bottom[2].data[i,:])
        	estimate = np.squeeze(bottom[3].data[i,:])
        	ind = np.argmax(estimate, axis=0)
        	print(ind)
        	estimateImg = ind.copy()
        	for j in range(0,estimate.shape[0]):
        		estimateImg[ind == j] = j
       		# ref = np.transpose(ref, (1,2,0))
       		# im = np.transpose(im, (1,2,0))
	        print(ref.shape)
	        print(label.shape)
	        print(np.unique(label))
	        plt.figure()
	        plt.imshow(ref,vmin=0, vmax=1)
	        plt.figure()
	        plt.imshow(im,vmin=0, vmax=1)
	        plt.figure()
	        plt.imshow(label,vmin=0, vmax=5)
	        plt.figure()
	        plt.imshow(estimateImg,vmin=0, vmax=5)
	        plt.show()
	        # plt.draw()
	        # plt.pause(0.001)
        	# input("Press [enter] to continue.")

    def backward(self, top, propagate_down, bottom):
        pass
        