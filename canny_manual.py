# encoding: utf-8
# Canny

import cv2
import cv2.cv
import numpy as np
import matplotlib.pyplot as plt
import math,os
from modules.classification import *
from modules.ccode import *

img=0
pos_p1 =4
pos_p2 = 4
pos_p3 = 3


def Process():

    global pos_p3
    if pos_p3%2==0:pos_p3=pos_p3+1

    img_=cv2.bilateralFilter(img,5,200,50)
    #img_=img
    detected_edges = cv2.Canny(img_,pos_p1,pos_p2,apertureSize=pos_p3)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)

    cv2.namedWindow('Process window',cv2.cv.CV_WINDOW_NORMAL)
    cv2.imshow('Process window', dst)

    #cv2.namedWindow('canny demo',cv2.cv.CV_WINDOW_NORMAL)
    #cv2.imshow('canny demo',dst)



def cb_p1(pos):
    global pos_p1
    pos_p1 = pos
    Process()

def cb_p2(pos):
    global pos_p2
    pos_p2 = pos
    Process()


def cb_p3(pos):
    global pos_p3
    pos_p3 = pos
    Process()

if __name__ == '__main__':

    dataSets = ['mnist_basic','mnist_noise_variation','mnist_background_images','mnist_background_random','mnist_rotation','mnist_rotation_back_image','mnist_rotation_new']
    featureExtrs = ['HOG''Gabor_set','PCA','ChainBlur']


    m=Classification()
    # set Parameters
    m.pixel = 28 # pixel size (only for a square)
    noiselevel=1
    m.dir_input = "..\data\input"
    m.dir_output = "..\data\output"
    m.dataName =dataSets[2]
    m.trainSize=m.valSize=m.teSize=200 #50000
    m.numEx = int(time.time())

    print 'No. %d starts'%m.numEx
    #################################
    # output file setting
    if 'noise' in m.dataName:m.dataName=m.dataName+'s_all_'+str(noiselevel)

    # set variables
    img_tr_all,m.lab_tr_all,img_vl_all,m.lab_vl_all,img_te_all,m.lab_te_all \
        = load(m.dir_input,m.dataName,m.pixel,trlen=m.trainSize,telen=m.teSize,vlen=m.valSize)


    #read an image
    #img = cv2.imread('01_cut.tif')
    dir_input = "..\data\input"
    img_ =img_tr_all[0].reshape(m.pixel,m.pixel)
    showimg(img_)

    img_ =(img_tr_all[0]*255).reshape(m.pixel,m.pixel)
    #src_f = np.array(src, dtype=np.float32)

    #create a gray scale version of the image, with as type an unsigned 8bit integer
    img = np.zeros( (img_.shape[0], img_.shape[1]), dtype=np.uint8 )
    img[:,:] = img_[:,:]

    #img = cv2.imread(os.path.join(dir_input,'example3.png'),1)

    cv2.namedWindow('Process window',cv2.cv.CV_WINDOW_NORMAL)
    cv2.createTrackbar('hysteresis th1','Process window',pos_p1,100,cb_p1)
    cv2.createTrackbar('hysteresis th2', 'Process window', pos_p2, 100, cb_p2)
    cv2.createTrackbar('aperture','Process window',pos_p3,100,cb_p3)

    Process()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


















