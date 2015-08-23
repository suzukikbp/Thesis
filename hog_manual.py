# encoding: utf-8

import os,time,csv,copy,pylab
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from modules.ccode import *


from skimage.feature import hog
from skimage import data, color, exposure,feature


src_f = 0
pos_cells =4
pos_pics = 4
pos_ori = 4
pics=0
nb_c=1

def Process():

    nb_c=pics/pos_pics
    if nb_c<pos_cells:
        pos_cells_local=nb_c
    else:pos_cells_local=pos_cells

    d=pos_ori*(pos_cells_local*pos_cells_local)*(nb_c-pos_cells_local+1)*(nb_c-pos_cells_local+1)
    print 'D:%d, nb_c:%d, pixs:%d, cells:%d(%d), orien:%d'%(d,nb_c,pos_pics,pos_cells_local,pos_cells,pos_ori)

    fd, hog_image = hog(src_f.reshape(pics,pics), orientations=pos_ori, pixels_per_cell=(pos_pics, pos_pics),cells_per_block=(pos_cells_local,pos_cells_local ), visualise=True)
    h = hog(src_f.reshape(pics,pics), orientations=pos_ori, pixels_per_cell=(pos_pics, pos_pics),cells_per_block=(pos_cells_local, pos_cells_local))
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    hog_image_rescaled = hog_image_rescaled.reshape(pics,pics)

    cv.namedWindow('Process window',cv.cv.CV_WINDOW_NORMAL)
    cv.namedWindow('Hog_image_rescaled',cv.cv.CV_WINDOW_NORMAL)
    cv.imshow('Process window', hog_image)
    cv.imshow('Hog_image_rescaled', hog_image_rescaled)

def cb_pics(pos):
    global pos_pics
    pos_pics = pos
    Process()

def cb_cells(pos):
    global pos_cells
    pos_cells = pos
    Process()


def cb_ori(pos):
    global pos_ori
    pos_ori = pos
    Process()


if __name__ == '__main__':
    dir_input = "..\data\input"
    image = cv.imread(os.path.join(dir_input,'example4_.png'),1)
    pics=60
    image = cv.resize(image, (pics, pics))
    cv.imshow('Src',image)
    src = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #global src_f
    src_f = np.array(src, dtype=np.float32)

    cv.namedWindow('Process window',cv2.cv.CV_WINDOW_NORMAL)
    #cv.createTrackbar('Sigma','Process window',pos_sigma,pos_ks,cb_sigma)
    cv.createTrackbar('pixels per cell','Process window',2,pics/3,cb_pics)
    cv.createTrackbar('cells per block', 'Process window', pos_cells, 100, cb_cells)
    cv.createTrackbar('orientation','Process window',pos_ori,100,cb_ori)

    Process()
    cv.waitKey(0)
    cv.destroyAllWindows()
