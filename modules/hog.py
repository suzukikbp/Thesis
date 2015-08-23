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

from skimage.feature import hog
from skimage import data, color, exposure,feature


class Hog():
    def main(self,pixel,img_tr,img_te,img_vl,dir_in,dir_out,bname,**kargs):

        self.SHOW = 0
        self.DEBUG = kargs['DEBUG']
        self.pixs = kargs['pixs']
        self.cells = kargs['cells']
        self.orientation = kargs['orientation']
        val= kargs['VAL']

        # 1. Set initial Parameters
        self.dir_input,self.dir_output,self.bname = dir_in,dir_out,bname
        self.img_tr,self.img_te,self.img_vl=img_tr,img_te,img_vl
        self.pixel=pixel
        self.results=[str(dir_out.split('\\')[-1])]
        tt=time.time()

        # file setting
        csvname=os.path.join(os.path.dirname(dir_out),'hogResults_'+bname+'.csv')
        self.f = open(csvname, 'ab+')
        self.csvWriter = csv.writer(self.f)

        # exception of negative valuable1: #cells/block
        if self.pixel/3<self.cells: return None

        # exception of negative valuable1: #pixels/cell
        block=self.pixel/self.pixs
        if block<self.cells: return None

        print '    D:%d, #c:%d, pixs:%d, cells:%d, orien:%d'%(kargs['dimention'],block,self.pixs,self.cells,self.orientation)


        if val:
            datas=[self.calcHog(imgs) for imgs in [self.img_tr,self.img_vl,self.img_te]]
        else:
            datas=[self.calcHog(imgs) for imgs in [self.img_tr,self.img_te]]
            datas.append(None)

        # Export characteristics of image
        self.exportImg(img_tr[0],bname)

        # Export results
        self.results.extend([time.time()-tt])
        params=[kargs['dimention'],block,self.pixs,self.cells,self.orientation]
        self.results.extend(params)
        self.csvWriter.writerow(self.results)
        self.f.close()
        del self.csvWriter,self.f



        return datas+[params]

    def calcHog(self,imgs):
        hogs=[]
        for img in imgs:
            hogs.append(hog(img.reshape(self.pixel,self.pixel), \
                orientations=self.orientation, pixels_per_cell=(self.pixs, self.pixs),\
                cells_per_block=(self.cells, self.cells)))
            #showimg(hog_image.reshape(28,28))
        return np.array(hogs)

    def exportImg(self,img,bname):
        if 'basic' in bname:img = img.T

        img=img.reshape(self.pixel,self.pixel).transpose()
        fd, hog_image = hog(img, \
            orientations=self.orientation, pixels_per_cell=(self.pixs, self.pixs),\
            cells_per_block=(self.cells, self.cells),visualise=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.axis('off')
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        ax2.axis('off')
        ax2.imshow(hog_image, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.savefig(os.path.join(self.dir_output,self.bname+'_Hog_p%d_c%d'%(self.pixs,self.cells)))
