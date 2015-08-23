# encoding: utf-8
import numpy as np
import os,cv2,time,csv,warnings
import matplotlib as pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage as nd
from scipy import misc as scm

from modules.ccode import *
from modules.pca import *
from modules.preposs import *
from modules.gabor import *
from modules.classification import *


class Gabor_opencv(Gabor):
    def __init__(self,pixel,img_tr,img_te,img_vl,dir_in,dir_out,dir_data,bname,**kargs):
        super(Gabor_opencv, self).__init__(pixel,img_tr,img_te,img_vl,dir_in,dir_out,dir_data,bname,**kargs)


    def buildKernel(self,ks, sigma, phi ,lamda,gamma=1):
        ker = cv2.getGaborKernel((ks, ks), sigma, phi, lamda, gamma, 0, ktype=cv2.CV_32F)
        ker /= 1.5*ker.sum()
        return ker
