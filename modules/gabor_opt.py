# encoding: utf-8
import numpy as np
import os,cv2,time,csv
import matplotlib.pyplot as plt
from scipy import ndimage as nd

import optunity,optunity.metrics

from modules.ccode import *
from modules.pca import *
from modules.preposs import *
from modules.gabor import *

SHOW = 0
DEBUG = 0

class Gabor_opt(Gabor):
#####################################################################
    def gaborMain(self):
#####################################################################
        # 1. Optimiza parameters for a kernel
        optparsL=[]
        optvals=[]
        for w in range(1,self.maxWidth+1):
            self.lamda = 2.*w
            # estimate range of sigma
            print '    Build several kernel'
            phi_d_max = np.max(self.phis)
            sinval=np.sin(phi_d_max/2.)
            sig1=self.lamda/(sinval*4.*np.sqrt(2)*np.pi)
            sig2=w*np.sqrt(2)
            sigmas = np.linspace(min(sig1,sig2),max(sig1,sig2),10)
            #################################
            nfol=2
            nitr=1
            @optunity.cross_validated(x=self.img_tr,num_folds=nfol, num_iter=nitr)
            def gabor_similarity(x_train,x_test,sigma,ksize):
                ksize = int(ksize)
                if not ksize%2: ksize += 1
                ecc,imgs = self.compareGabor(sigma=sigma,ksize=ksize,lamda=self.lamda,imgs=x_train)
                return ecc/len(x_train)
            kwargs = {'sigma':[min(sigmas),max(sigmas)],'ksize':[3,25]}

            optpars, optpars2, optpars3 = optunity.minimize(gabor_similarity,**kwargs)
            ksizes=np.array(optpars2.call_log['args']['ksize'])
            sigs = np.array(optpars2.call_log['args']['sigma'])
            vals = np.array(optpars2.call_log['values'])
            self.plotScatter([[ksizes.astype(int),vals]],'K size','Entropy correlation coefficient','Opt wrt ksize when w '+str(w),dir_o=self.dir_output,tnam='Width=%d'%w)
            self.plotScatter([[sigs,vals]],'Sigma','Entropy correlation coefficient','Opt wrt sigma when w '+str(w),dir_o=self.dir_output,tnam='Width=%d'%w)
            #self.plot3D([[ksizes.astype(int),sigs,vals]],'K size','Sigma','Entropy correlation coefficient','Opt when w '+str(w),dir_o=self.dir_output,tnam='Width=%d'%w)

            opts=np.array([ksizes,sigs])
            idx = np.where((opts[0]==optpars['ksize'])+(opts[1]==optpars['sigma']))[0][0]
            optvals.append(vals[idx])
            optparsL.append(optpars)

        #the index of w which got the smallest value
        idx = np.where(optvals==min(optvals))[0][0]
        self.sigma=optparsL[idx]['sigma']
        self.ksize=optparsL[idx]['ksize']



        # 3. Adaptive regulation of outputs of Gabor filters
        print '    Adaptive regulation'
        tt=time.time()
        alphas = np.linspace(0,5,10)
        if DEBUG ==1:alphas = np.linspace(0,10,10)
        self.lamdas = np.linspace(0.5,self.lamda,self.lamda*2)
        nfol=2
        nitr=1
        @optunity.cross_validated(x=self.img_tr, num_folds=nfol, num_iter=nitr)
        def alpha_loss(x_train,x_test,alpha):
            rmses_lam,maxOuts_lam,_,_=self.lossFun_alpha(lamdas=self.lamdas,alpha=alpha,imgs=x_train)
            print '     alpha: %.3f, rmse_sum: %.3f'%(alpha,rmses_lam.sum())
            return rmses_lam.sum()
        kwargs = {'alpha':[min(alphas),max(alphas)]}
        optpars, optpars2, optpars3 = optunity.minimize(alpha_loss,**kwargs)
        self.alpha=optpars['alpha']
        self.results.extend([optpars3['solver_name'],str(optpars3['num_generations']),str(optpars3['num_particles']),str(nfol),str(nitr),str(round(time.time()-tt,3))])

        #################################
        # 4. Set Gabor kernel with optimal params
        self.kers = [] # kernels (each phi)
        for phi in self.phis:
            self.kers.append(self.buildKernel(self.ksize,self.sigma, phi ,self.lamda))
        self.kers=np.array(self.kers)
        print '    Optimal params'
        print '     ksize: %d, sig: %.2f,lamda: %.1f, alpha: %.2f'%(self.ksize,self.sigma,self.lamda,self.alpha)

        #################################
        # 5. Produce output image and extract features: local histgram
        self.blocksize = 5
        self.tau = 6
        brange =self.blocksize/2
        # train data
        print '    Feature selection train'
        hist_tr = self.extractFeatures(self.img_tr,brange)
        # test data
        print '    Feature selection test'
        hist_te = self.extractFeatures(self.img_te, brange)

        #################################
        # 6. Compress dimentionality : PCA
        [eigenvalues, eigenvectors, mu] = pca(hist_tr)
        # select #CP
        cp=show_eigenvalues(eigenvalues,self.dir_output,name=self.bname,ratio=0.90)
        print '    PCA: %d'%cp
        eigenvalues,eigenvectors=select_eigenvalues(eigenvalues,eigenvectors,num_components=cp)
        # projection
        xtrain = pca_project(eigenvectors,hist_tr,mu)
        xtest = pca_project(eigenvectors,hist_te,mu)

        #################################
        # 7. Export results and return
        self.results.extend([str(self.maxWidth),str(self.ksize),str(self.sigma),str(self.lamda),str(len(self.phis)),str(self.alpha),str(self.blocksize),str(self.tau),str(cp)])
        self.csvWriter.writerow(self.results)
        self.f.close()
        del self.csvWriter,self.f

        params = '%s;%s;%s;%s;%s;%s;%s;%s;%s'%(str(self.maxWidth),str(self.ksize),str(self.sigma),str(self.lamda),str(len(self.phis)),str(self.alpha),str(self.blocksize),str(self.tau),str(cp))
        return [xtrain, xtest,params]










#####################################################################
    def __init__(self,pixel,img_tr,img_te,dir_input, dir_output,bname,w,ksize=21):
#####################################################################
        # set Parameters
        self.dir_input = dir_input
        self.dir_output= dir_output
        self.bname = bname
        self.img_tr=img_tr
        self.img_te=img_te
        #self.lab_tr=lab_tr
        self.pixel=pixel

        self.maxWidth = int(w) #width = 4. #4. # stroke width
        #self.ksize = ksize # kernel size
        self.phis = np.array([-np.pi/2,-np.pi/4,0,np.pi/4]) # orientation

        csvname=os.path.join(os.path.dirname(dir_output),'gabourResults_opt.csv')
        self.f = open(csvname, 'ab+')
        self.csvWriter = csv.writer(self.f)
        self.results=[str(dir_output.split('\\')[-1])]






