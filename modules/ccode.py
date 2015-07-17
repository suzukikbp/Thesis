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
import matplotlib.pyplot as plt


def showimg(img):
    cv2.namedWindow('img',cv2.cv.CV_WINDOW_NORMAL)
    cv2.imshow('img',img)
    cv2.waitKey(0)

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
        #showing(imgxy*180)
    return np.array(imgs_).reshape(len(imgs_),pixel^2)

def contour2img(cont):
    img=np.zeros([pixel,pixel])
    for i in range(0,cont.shape[0]):
        x=cont[i][0][0]
        y=cont[i][0][1]
        img[y,x]=1
    #showimg(img)
    return img

# make Directions from contour images
#   cont : the array of a contour
#   pixel: the size of image (int)
def img2direct(img):
    a0=np.zeros([pixel,pixel])
    a1=np.zeros([pixel,pixel])
    a2=np.zeros([pixel,pixel])
    a3=np.zeros([pixel,pixel])
    a4=np.zeros([pixel,pixel])
    a5=np.zeros([pixel,pixel])
    a6=np.zeros([pixel,pixel])
    a7=np.zeros([pixel,pixel])

    for y in range(1,pixel-1):
        for x in range(1,pixel-1):
            if (img[y,x]==1):
                if(img[y,x+1]==1):a0[y,x]=1         #0: right
                if(img[y+1,x+1]==1):a1[y,x]=1       #1: upperright
                if(img[y+1,x]==1):a2[y,x]=1         #2: above
                if(img[y+1,x-1]==1):a3[y,x]=1       #3: upperleft
                if(img[y,x-1]==1):a4[y,x]=1         #4: left
                if(img[y-1,x-1]==1):a5[y,x]=1       #5: lowerleft
                if(img[y-1,x]==1):a6[y,x]=1         #6: below
                if(img[y-1,x+1]==1):a7[y,x]=1       #7: lowerright
    direct = np.array([a0,a1,a2,a3,a4,a5,a6,a7])
    #for i in range(0,8):showimg(direct[i])
    return direct

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
        gaussian.append(rimg)
    return gaussian

# make Chaincode for multiple images
#   thresh : the array of images
#   pixel: the size of image (int)
def chaincode(img,pix,dir):
    global dir_out
    dir_out=dir
    global pixel
    pixel = pix

    def showContour(img,cont):
        tempimg = img.reshape(pixel,pixel)*180
        cv2.drawContours(tempimg, cont, -1, (128,255,128), -1 )
        cv2.namedWindow("Show Image",cv2.cv.CV_WINDOW_NORMAL)
        cv2.imshow("Show Image",tempimg)
        cv2.waitKey(0)

    def showImgs(im,name):
        if(len(plt.get_fignums())>0):plt.close()
        for i in range(0,8):
            plt.subplot(1, 8, i+1)
            plt.imshow(im[i],plt.cm.binary, interpolation='none')
            plt.title(str(i))
            plt.rcParams['font.size'] = 8
        plt.savefig(os.path.join(dir_out,name+'.png'))

    conts=[]
    directs=[]
    hierarchys = []
    gaussians = []

    # convert into binary image
    _,thresh = makeBinary(img)

    for i in range(0,thresh.shape[0]):
        img =thresh.copy()[i].reshape(pixel,pixel)
        cont, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        conts.append(cont)
        hierarchys.append(hierarchy)
        # extract directions from contours
        img=contour2img(cont[0])
        direct=img2direct(img)
        # convolution and sampling the directional sub-images
        # sampling from directctions
        gaussians.append(direct2gaussian(direct,pixel,5))
        # drawing
        #showContour(thresh[i],cont[0])
        #showContour(np.zeros([pixel,pixel]),cont[0])
        #showImgs(direct,'Direction')
        #showImgs(direct2gaussian(direct,pixel,5),'Sampling')

    # reshape for feature matrix
    gaussians = np.array(gaussians)
    gaussians = gaussians.reshape(gaussians.shape[0],gaussians.shape[1]*gaussians.shape[2]*gaussians.shape[3])

    return np.array(gaussians),conts

