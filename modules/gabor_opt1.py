# encoding: utf-8
import numpy as np
import os,cv2,time,csv
import matplotlib as pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage as nd

import optunity,optunity.metrics,optunity.solvers

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

solvers = ['particle swarm','grid search','random search','cma-es','nelder-mead']

class Gabor_opt1(Gabor):
    def __init__(self,pixel,img_tr,img_te,img_vl,dir_in,dir_out,bname,\
                ws=[5],lms=[1],sigmas=[5],nbPhis=[4],ksizes=[3],ns=[7],
                #numg=3,nump=4,nfol=2,nitr=1):
                optparms=[3,4,2,1]):

        self.SHOW = 0
        self.DEBUG = 1
        self.EXP = True
        self.width=-1
        if self.DEBUG==1:print 'DEBUG MODE'

        # 1. Set initial Parameters
        self.dir_input,self.dir_output,self.bname = dir_in,dir_out,bname
        self.img_tr,self.img_te,self.img_vl=img_tr,img_te,img_vl
        self.pixel=pixel
        numg,nump,nfol,nitr=optparms[0],optparms[1],optparms[2],optparms[3]
        self.results=[str(dir_out.split('\\')[-1])]
        csvname=os.path.join(os.path.dirname(dir_out),'gabourResults.csv')
        self.f = open(csvname, 'ab+')
        self.csvWriter = csv.writer(self.f)

        # 2. Optimize parameters
        #numg,nump=3,4
        #nfol,nitr=2,1
        @optunity.cross_validated(x=self.img_tr, num_folds=nfol, num_iter=nitr)
        def set_sigma(x_train,x_test,lamda,nbPhi,n,ksize,sigma):
            self.exportImgs=[]
            # modify params
            lamda /=100.
            sigma /=10.
            nbPhi=int(nbPhi)
            ksize,n,D,nbPhi=self.modifyParam(ksize,n,nbPhi)
            phis = np.linspace(-np.pi/2,np.pi/4,nbPhi)
            # calculate ecc
            ecc,imgs = self.compareGabor(sigma=sigma,ksize=ksize,lamda=lamda,phis=phis,N=n,imgs=x_train)
            tnam='    w:_,lamda:%.2f ,#phis:%d, ksize:%d, D:%d(N:%d),sig:%.3f,AECCS:%.3f'%(lamda,int(nbPhi),ksize,D,n,sigma,ecc/len(x_train))
            print tnam
            if ecc/len(x_train)<0.1:print 'error_AECCS'

            if self.EXP:
                fn ='W%sL%sNP%sK%sD%s'%('_',str(int(round(lamda,2)*100)).zfill(4),str(nbPhi).zfill(2),str(ksize).zfill(2),str(D).zfill(2))
                lab = ['' for i in range(len(self.exportImgs))]
                ncol=nbPhi*2+1
                drawimg(self.exportImgs,lab,self.dir_output,self.bname+'_ker_opt2'+fn,\
                    self.pixel,ncol=ncol,nrow=1,skip=False,tnam=tnam)
            return ecc/len(x_train)

        solver = optunity.make_solver(solvers[0],\
                    lamda=[int(min(lms)),int(max(ws))],\
                    nbPhi=[int(min(nbPhis)),int(max(nbPhis))],\
                    n=[int(min(ns)),int(max(ns))],\
                    ksize=[int(min(ksizes)),int(max(ksizes))],\
                    sigma=[int(min(sigmas)),int(max(sigmas))],\
                    num_particles=nump, num_generations=numg)
        optpars, optpars2 = optunity.optimize(solver,set_sigma,maximize=False)

        # 3. Set optimal parameters
        self.ksize,self.N,self.D,self.nbPhi=self.modifyParam(optpars['ksize'],optpars['n'],optpars['nbPhi'])
        self.lamda,self.sigma=optpars['lamda']/100.,optpars['sigma']/10.
        self.phis=np.linspace(-np.pi/2,np.pi/4,self.nbPhi)
        self.eccs=min(np.array(optpars2.call_log['values']))
        # export
        print '    === Optimap params ===\n    w:_, lm:%.2f, #phis:%d, ksize:%d, D:%d(N:%d),sig:%.3f,AECCS:%.3f'%(self.lamda,self.nbPhi,self.ksize,self.D,self.N,self.sigma,self.eccs)
        self.results.extend([solvers[0],str(numg),str(nump),str(nfol),str(nitr),str(optpars2.stats['time'])])

        # 4. Set optimal param for additional regulation
        self.setAlpha(numg=numg,nump=nump,nfol=nfol,nitr=nitr)

        #  Export results
        self.results.extend([str(self.width),str(self.ksize),str(self.sigma),str(self.lamda),str(len(self.phis)),str(self.alpha),str(self.blocksize),str(self.tau),str(cp),str(self.eccs)])
        self.csvWriter.writerow(self.results)
        self.f.close()
        del self.csvWriter,self.f

# Adaptive regulation of outputs of Gabor filters
#####################################################################
    def setAlpha(self,numg=2,nump=4,nfol=2,nitr=1):
#####################################################################
        print '  Adaptive regulation'
        # 1. Set initial Parameters
        #alphas = np.linspace(0,5,10)
        alphas = np.linspace(1,5,10)
        if self.DEBUG ==1:alphas = np.linspace(1,5,10)
        #self.lamdas = np.linspace(0.001,self.lamda,self.lamda*2)
        self.lamdas = np.linspace(0.001,self.lamda,10)

        # 2. Optimize parameters
        @optunity.cross_validated(x=self.img_tr, num_folds=nfol, num_iter=nitr)
        def alpha_loss(x_train,x_test,alpha):
            if alpha < 1:alpha=1
            rmses_lam,maxOuts_lam,_,_=self.lossFun_alpha(lamdas=self.lamdas,alpha=alpha,imgs=x_train)
            print '    alpha: %.3f, rmse_sum: %.3f'%(alpha,rmses_lam.sum())
            return rmses_lam.sum()

        solver = optunity.make_solver(solvers[0],alpha=[min(alphas),max(alphas)],\
                    num_particles=nump, num_generations=numg)
        optpars, optpars2 = optunity.optimize(solver,alpha_loss,maximize=False)

        # 3. Set parameters
        self.alpha=optpars['alpha']
        self.results.extend([solvers[0],str(numg),str(nump),str(nfol),str(nitr),str(optpars2.stats['time'])])
        print '  Adaptive regulation:%.2fsec'%(optpars2.stats['time'])





