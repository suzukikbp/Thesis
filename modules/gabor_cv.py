# encoding: utf-8
import numpy as np
import os,cv2,time,csv
import matplotlib as pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage as nd

from modules.ccode import *
from modules.pca import *
from modules.preposs import *
from modules.gabor import *

# width(w): width of stroke = kernel size
# lamda: wavelength of simusdidal factor (scale)
# ksize: kernel size
# phis: orientation of the stroke
# nbPhi: # of  phis
# N: # of block in the images
# D: # of sampling interval (# pixel)
# sigma:


class Gabor_cv(Gabor):
    #def __init__(self,pixel,img_tr,img_te,dir_in,dir_out,bname,\
    #            w=-1,sigma=-1,nbPhi=-1,ksize=-1,n=-1,d=-1,alpha=-1):
    def __init__(self,g):

        self=g
        self.SHOW = 0
        self.DEBUG = 1
        self.EXP = True

        # Initialization
        self.exportImgs=[]

        # 1. Set initial Parameters
        #self.dir_input,self.dir_output,self.bname = dir_in,dir_out,bname
        #self.img_tr,self.img_te=img_tr,img_te
        #self.pixel=pixel
        #self.results=[str(dir_out.split('\\')[-1])]
        csvname=os.path.join(os.path.dirname(self.dir_output),'gabourResults.csv')
        self.f = open(csvname, 'ab+')
        self.csvWriter = csv.writer(self.f)

        # 3. Set optimal parameters
        #self.ksize,self.N,self.D,self.width,self.sigma,self.nbPhi,self.alpha\
        #    =ksize,n,d,ww,sigma,nbPhi,alpha
        self.phis=np.linspace(-np.pi/2,np.pi/4,self.nbPhi)
        #self.lamda = 2.*self.width

        #for i in range(0,len(alphas)):
        #    lab[i]='alpha:%.1f'%alphas[i]
        ecc,imgs = self.compareGabor(sigma=self.sigma,ksize=self.ksize,lamda=self.lamda,phis=self.phis,N=self.N,imgs=self.img_tr)
        lab = ['' for i in range(0,len(self.exportImgs))]
        drawimg(self.exportImgs,lab,self.dir_output,self.bname+'_ker_cv',self.pixel,ncol=self.nbPhi*2+1,nrow=1,skip=False,tnam='alpha=%.2f'%self.alpha)
        rmses_lam,maxOuts_lam,gbr,gbr_ad=self.lossFun_alpha(lamdas=self.lamdas,alpha=self.alpha)
        drawimg([gbr,gbr_ad],['',''],self.dir_output,self.bname+'_alpha_cv',self.pixel,ncol=2,nrow=1,skip=False)


        print 'Gabor_cv'


    def buildKernel(self,ks, sig, phi ,lamda):
        params = {'ksize':ks, 'sigma':sig, 'theta':phi, 'lambd':lamda,
                  'gamma':1, 'psi':0, 'ktype':cv2.CV_32F}
        ker = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        return ker

    #https://gist.github.com/odebeir/3918044