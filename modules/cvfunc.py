#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      KSUZUKI
#
# Created:     14/02/2015
# Copyright:   (c) KSUZUKI 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import os,cv2,time
import numpy as np


def makeBinary(img_tr):
    img_cvt_tr = (img_tr*255).astype(np.uint8)
    ret_tr,thresh_tr = cv2.threshold(img_cvt_tr.copy(),5,1,0)
    return ret_tr,thresh_tr


# make Sobel for multiple images
#   thresh : the array of images
#   pixel: the size of image (int)
def makeSober(imgs,pixel):
    imgs_=[]
    for i in range(0,imgs.shape[0]):
        img =imgs.copy()[i].reshape(pixel,pixel)
        imgy = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
        imgx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
        imgxy = cv2.add(imgx,imgy)
        imgs_.append(imgxy)


        # drawing
        #cv2.imshow("Show Image",imgxy*180)
        #cv2.waitKey(0)


    #return np.array(directs)
    return np.array(imgs_)



# make Contour for multiple images
#   thresh : the array of images
#   pixel: the size of image (int)
def makeContor(thresh,pixel):
    conts=[]
    directs=[]
    hierarchys = []
    gaussians = []
    for i in range(0,thresh.shape[0]):
        img =thresh.copy()[i].reshape(pixel,pixel)
        cont, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        conts.append(cont)
        hierarchys.append(hierarchy)
        #directs.append(contor2direct(cont,pixel))
        direct =contor2direct(cont,pixel)
        gaussians.append(direct2gaussian(direct,pixel,5))
        """
        # drawing
        tempimg = thresh[0].reshape(pixel,pixel)*180
        cv2.drawContours(tempimg, cont[0], -1, (128,255,128), -1 )
        cv2.namedWindow("Show Image",cv2.cv.CV_WINDOW_NORMAL)
        cv2.imshow("Show Image",tempimg)
        cv2.waitKey(0)
        """

    #return thresh,np.array(conts), np.array(hierarchys), np.array(directs)
    #return thresh,np.asarray(conts), np.asarray(hierarchys), np.asarray(directs)
    # reshape
    gaussians = np.array(gaussians)
    gaussians = gaussians.reshape(gaussians.shape[0],gaussians.shape[1]*gaussians.shape[2]*gaussians.shape[3])

    #return np.array(directs)
    return np.array(gaussians)


# make Directions from contour images
#   cont : the array of a contour
#   pixel: the size of image (int)
def contor2direct(cont,pixel):
    #a0=a1=a2=a3=a4=a5=a6=a7=np.zeros([pixel,pixel])
    a0=np.zeros([pixel,pixel])
    a1=np.zeros([pixel,pixel])
    a2=np.zeros([pixel,pixel])
    a3=np.zeros([pixel,pixel])
    a4=np.zeros([pixel,pixel])
    a5=np.zeros([pixel,pixel])
    a6=np.zeros([pixel,pixel])
    a7=np.zeros([pixel,pixel])

    for i in range(0,len(cont)):
        cxy = cont[i][0]
        for j in range(1,len(cont[i])):
            nxy = cont[i][j]
            cx = cxy[0][0]
            cy = cxy[0][1]
            nx = nxy[0][0]
            ny = nxy[0][1]
            # left collumn
            if((cx-nx)==1):
                if((cy-ny)==1):
                    a6[nx][ny]=1
                elif((cy-ny)==0):
                    a7[nx][ny]=1
                else:
                    a0[nx][ny]=1
            # center collumn
            elif((cx-nx)==0):
                if((cy-ny)==1):
                    a5[nx][ny]=1
                else:
                    a1[nx][ny]=1
            # right collumn
            else:
                if((cy-ny)==1):
                    a4[nx][ny]=1
                elif((cy-ny)==0):
                    a3[nx][ny]=1
                else:
                    a2[nx][ny]=1
            cxy=nxy

    return np.array([a0,a1,a2,a3,a4,a5,a6,a7])


# Filter and sample directional images
#   cont : the array of a contour
#   pixel: the size of image (int)
def direct2gaussian(directs,pixel,filter_length):

    gaussian = []
    for j in range(0,directs.shape[0]):
        img = directs[j]
        # convolution with gaussian filter
        gaus = cv2.GaussianBlur(img,(filter_length,filter_length),0)
        # sampling the filtered image
        rimg = cv2.resize(gaus, (filter_length,filter_length),interpolation =cv2.cv.CV_INTER_CUBIC)
        #rimg = cv2.resize(directs[i][j], (filter_length,filter_length),interpolation =cv2.cv.CV_INTER_NN)
        gaussian.append(rimg)

    return gaussian



"""
# Filter and sample directional images
#   cont : the array of a contour
#   pixel: the size of image (int)
def direct2gaussian(directs,pixel,filter_length):

    gaussians = []
    for i in range(0,directs.shape[0]):
        gaussian = []
        for j in range(0,directs.shape[1]):
            img = directs[i][j]
            # convolution with gaussian filter
            gaus = cv2.GaussianBlur(img,(filter_length,filter_length),0)
            # sampling the filtered image
            rimg = cv2.resize(gaus, (filter_length,filter_length),interpolation =cv2.cv.CV_INTER_CUBIC)
            #rimg = cv2.resize(directs[i][j], (filter_length,filter_length),interpolation =cv2.cv.CV_INTER_NN)
            gaussian.append(rimg)
        gaussians.append(gaussian)

    # reshape
    gaussians = np.array(gaussians)
    gaussians = gaussians.reshape(gaussians.shape[0],gaussians.shape[1]*gaussians.shape[2]*gaussians.shape[3])

    return gaussians

"""



