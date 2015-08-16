# encoding: utf-8

#-------------------------------------------------------------------------------
# Name:        hog
# Purpose:
#
# Author:      KSUZUKI
#
# Created:     15/08/2015
# Copyright:   (c) KSUZUKI 2015
# Licence:     http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
#-------------------------------------------------------------------------------

import os,cv2,time,csv,copy,pylab
import numpy as np

class Hog():
    def main(self,train_cells,test_cells):
        self.SZ=20
        self.bin_n = 16 # Number of bins
        self.affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
        #cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

        # First half is trainData, remaining is testData
        #train_cells = [ i[:50] for i in cells ]
        #test_cells = [ i[50:] for i in cells]


        deskewed = [map(deskew,row) for row in train_cells]
        hogdata = [map(hog,row) for row in deskewed]
        trainData = np.float32(hogdata).reshape(-1,64)
        responses = np.float32(np.repeat(np.arange(10),250)[:,np.newaxis])

        """
        svm = cv2.SVM()
        svm.train(trainData,responses, params=svm_params)
        svm.save('svm_data.dat')

        ######     Now testing      ########################

        deskewed = [map(deskew,row) for row in test_cells]
        hogdata = [map(hog,row) for row in deskewed]
        testData = np.float32(hogdata).reshape(-1,self.bin_n*4)
        result = svm.predict_all(testData)

        #######   Check Accuracy   ########################
        mask = result==responses
        correct = np.count_nonzero(mask)
        print correct*100.0/result.size
        """

    def deskew(self,img):
        # calculate moment
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*self.SZ*skew], [0, 1, 0]])
        img = cv2.warpAffine(img,M,(self.SZ, self.SZ),flags=self.affine_flags)
        return img

    def hog(self,img):
        # Sobel derivatives of each cell in X and Y direction
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        # convert to the magnitude and direction of gradient at each pixel
        mag, ang = cv2.cartToPolar(gx, gy)
        #  This gradient is quantized to 16 integer values
        bins = np.int32(self.bin_n*ang/(2*np.pi))
        # Divide this image to four sub-squares
        bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        # For each sub-square, calculate the histogram of direction (16 bins)
        # weighted with their magnitude
        hists = [np.bincount(b.ravel(), m.ravel(), self.bin_n) for b, m in zip(bin_cells, mag_cells)]
        # a feature vector containing 64 values
        hist = np.hstack(hists)
        return hist


