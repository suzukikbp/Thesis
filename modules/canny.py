# encoding: utf-8

#-------------------------------------------------------------------------------
# Name:        hog
# Purpose:
#
# Author:      KSUZUKI
#
# Created:     15/08/2015
# Copyright:   (c) KSUZUKI 2015
# Licence:
#-------------------------------------------------------------------------------

import os,cv2,time,csv,copy,pylab
import numpy as np
import matplotlib.pyplot as plt

from modules.ccode import *
from modules.bilateral import *

from skimage.feature import hog
from skimage import data, color, exposure,feature


class Canny():
    def main(self,pixel,img_tr,img_te,img_vl,dir_in,dir_out,bname,**kargs):


        self.SHOW = 0
        self.DEBUG = kargs['DEBUG']
        self.diam = kargs['diameter_bil']
        self.sigCol = kargs['sigCol_bil']
        self.sigSpace = kargs['sigSpace_bil']
        self.th1 =kargs['th1_canny']
        self.th2 =kargs['th2_canny']
        self.ksize =kargs['ksize_canny']

        val= kargs['VAL']

        # 1. Set initial Parameters
        self.dir_input,self.dir_output,self.bname = dir_in,dir_out,bname
        self.img_tr,self.img_te,self.img_vl=img_tr,img_te,img_vl
        self.pixel=pixel
        self.results=[str(dir_out.split('\\')[-1])]
        tt=time.time()

        # file setting
        csvname=os.path.join(os.path.dirname(dir_out),'cannyResults_'+bname+'.csv')
        self.f = open(csvname, 'ab+')
        self.csvWriter = csv.writer(self.f)


        print '    Bilateral diameter:%d, sigCol:%d, sigSpace:%d'%(self.diam,self.sigCol,self.sigSpace)
        print '    Canny threhold1:%d, threhold2:%d, ksize:%d'%(self.th1,self.th2,self.ksize)

        # Bilateral filter and canny ditection
        if val:datas=[self.img_tr,self.img_te,self.img_vl]
        else:datas=[self.img_tr,self.img_te]
        cannys = [self.canny_process(imgs) for imgs in datas]

        # Compress dimentionality : PCA
        pcas = PCA().main(cannys,self.dir_output,self.bname)
        cp=pcas[-1]

        print '    PCA: %dcp'%cp

        # 4. Export results
        params=[self.diam,self.sigCol,self.sigSpace,self.th1,self.th2,self.ksize,cp]
        self.results.extend(params)
        self.csvWriter.writerow(self.results)
        self.f.close()
        del self.csvWriter,self.f

        # Export characteristics of image
        self.exportImg(img_tr[0],bname)


        if val:return pcas[:-1]+[params]
        else:return pcas[:-1]+[None,params]





    def canny_process(self,imgs):
        cannys=[]
        for img in imgs:
            # set bilateral filter
            img =(img*255).reshape(self.pixel,self.pixel)
            img_bil=cv2.bilateralFilter(img,self.diam,self.sigCol,self.sigSpace)
            # detect edges
            edges = cv2.Canny(img_bil.astype(np.uint8),self.th1,self.th2,apertureSize=self.ksize)
            dst = cv2.bitwise_and(img,img,mask=edges)
            cannys.append(dst.flatten())
        return np.array(cannys)


    def exportImg(self,img,bname):
        if 'basic' in bname:img = img.T
        img =(img*255).reshape(self.pixel,self.pixel)
        img_bil=cv2.bilateralFilter(img,self.diam,self.sigCol,self.sigSpace)
        # detect edges
        edges = cv2.Canny(img_bil.astype(np.uint8),self.th1,self.th2,apertureSize=self.ksize)
        dst = cv2.bitwise_and(img,img,mask=edges)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.axis('off')
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        ax2.axis('off')
        ax2.imshow(dst, cmap=plt.cm.gray)
        ax2.set_title('Bilateral+Canny')
        plt.savefig(os.path.join(self.dir_output,self.bname+'_Bil_d%dsc%dss%d_Can_th1_%dth2_%dks%d'%(self.diam,self.sigCol,self.sigSpace,self.th1,self.th2,self.ksize)))
