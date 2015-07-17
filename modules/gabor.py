# encoding: utf-8
import numpy as np
import os,cv2,time,csv
import matplotlib as pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage as nd

import optunity,optunity.metrics

from modules.ccode import *
from modules.pca import *
from modules.preposs import *

SHOW = 0
DEBUG = 0

class Gabor():
    def buildKernel(self,ks, sigma, phi ,lamda):
        # interval
        xs=ys=np.linspace(-ks/2.,ks/2.,ks)
        x,y = np.meshgrid(xs,ys)
        # rotation
        R1 = x*np.cos(phi)+y*np.sin(phi)
        R2 =-x*np.sin(phi)+y*np.cos(phi)

        exy = -0.5/(sigma**2)
        cscale = 2.*np.pi/lamda

        scale=1./(2*np.pi*(sigma**2))
        b =np.exp(exy*(R1**2+R2**2))
        c =np.exp(1j*(cscale*R1))
        ker = np.array(scale*b*c,dtype=np.complex128)

        ## algorithm of opencv
        #scale = 1.
        #b =np.exp(exy*(R1**2+R2**2))
        #c =np.cos(cscale*R1)
        #ker = np.array(scale*b*c,dtype=np.float32)

        return ker

    def showLineGraph(self,data1,data2,xlab,ylab):
        plt.clf()
        plt.plot(data1,data2)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        #plt.legend()
        plt.savefig(os.path.join(self.dir_output,self.bname+'_'+ylab+'.png'))

    # Compare output of Gabor filter with different phis
    #   @sigma: to build kernel
    #   @imgs: images to apply gabor filter
    #   @return ecc: entropy correlation coefficient
    #   @return imgs: output applied gabor filter
    def compareGabor(self,ksize=-1,lamda=-1,sigma=0,imgs=[]):
        if len(imgs)==0:imgs=self.img_tr
        if ksize==-1:ksize=self.ksize
        if lamda==-1:lamda=self.lamda
        ecc = 0.
        imgList=[]
        for img in imgs:
            ecc_img= 0.
            c=0
            test = 0
            for i in range(0,len(self.phis)): # iterate over phis
                for j in range(0,len(self.phis)):
                    if i>j:
                        c +=1
                        #print '     sig: %0.2f, phi:%0.2f'%(sig,phi)
                        keri =self.buildKernel(ksize, sigma, self.phis[i] ,lamda)
                        kerj =self.buildKernel(ksize, sigma, self.phis[j] ,lamda)
                        if SHOW == 1:self.showGFilter(keri)
                        # compare output of Gabor filter with different phis
                        gbr_i = cv2.filter2D(img.reshape(self.pixel,self.pixel), cv2.CV_32F,np.real(keri))
                        gbr_j = cv2.filter2D(img.reshape(self.pixel,self.pixel), cv2.CV_32F,np.real(kerj))
                        #showimg(gbr_i)
                        ecc_img += self.calcECC(gbr_i,gbr_j)
                        imgList.append(gbr_i)
            ecc += ecc_img/c
        return ecc,imgList

    # Calculate Entropy Correlation Coefﬁcient (ECC)
    def calcECC(self,img_i,img_j):

        # Normalization by Scaling Between 0 and 1
        def normalization(img,min_,max_):
            return np.array(((img-min_)/(max_-min_))*255,np.uint8)
        minVal = min(img_i.min(),img_j.min())
        maxVal = max(img_i.max(),img_j.max())
        img_i = normalization(img_i,minVal,maxVal)
        img_j = normalization(img_j,minVal,maxVal)

        # histgoram of images
        hist_i = np.bincount(img_i.ravel(),minlength=256)
        hist_j = np.bincount(img_j.ravel(),minlength=256)

        if(SHOW==1):
            plt.subplot(1,2,1);plt.plot(hist_i);
            plt.subplot(1,2,2);plt.plot(hist_j);
            plt.show()
        # calculate entropy
        hi = 0.0
        hj = 0.0
        hij = 0.0
        temp=0.
        sum_i = np.sum(hist_i)
        sum_j = np.sum(hist_j)
        if (sum_i==0 or sum_j==0):return 0 # zero devision
        for k in range(0,len(hist_i)):
            # probability
            pi= hist_i[k]*1./sum_i
            pj= hist_j[k]*1./sum_j
            pij = (hist_i[k]+hist_j[k])*1./(sum_i+sum_j)
            temp +=pi
            # entropy
            if(pi!=0):hi += pi*np.log(pi)
            if(pj!=0):hj += pj*np.log(pj)
            if(pij!=0):hij += pij*np.log(pij)
        # mutual information
        mutual_info = -hi-hj+hij
        # Entropy Correlation Coefﬁcient (ECC)
        if (-hi-hj)==0:return 0  # zero devision
        ecc = 2.0*mutual_info/(-hi-hj)
        return ecc

    # Addaptive regulation by modified sigmoid function
    #   @rimg: real part of the output
    def ad_Regulation(self,rimg,alpha,beta=0,chi=0):
        return np.tanh(alpha*rimg)  # original
        #return np.tanh(alpha*(rimg-chi))+beta  # modified version

    # Calculate selectivity
    #   @rimg: real part of the output
    def w_selectivity(self,rimg_org,rimg_ad):
        return (rimg_org.max()-rimg_ad.max())**2

    # Loss function to determine alpha
    #   @lamdas: the range of lamdas
    #   @return rmses_lam: rmses between gbr and gbr with addaptive regulation
    #   @return maxOuts_lam: max values of gbrs
    def lossFun_alpha(self,lamdas=[],alpha=1.,imgs=[],plot=False):
        if len(imgs)==0:imgs=self.img_tr
        rmses_lam=[]
        maxOuts_lam=[]
        for lam in lamdas:   # iteration ober lamdas
            rmse_lam=0.
            maxOut_lam=0.
            for phi in self.phis: # iteration over phis
                ker =self.buildKernel(self.ksize,self.sigma, phi ,lam)
                for img in imgs:
                    # output with gabor filter
                    gbr = cv2.filter2D(img.reshape(self.pixel,self.pixel), cv2.CV_32F,np.real(ker))
                    # output with gabor filter with regulatization
                    gbr_ad = self.ad_Regulation(np.real(gbr),alpha)
                    # rmse
                    rmse_lam += self.w_selectivity(gbr,gbr_ad)
                    maxOut_lam += np.max(gbr)
            rmse_lam = rmse_lam/(len(self.phis)*len(imgs))
            rmses_lam.append(rmse_lam)
            maxOuts_lam.append(maxOut_lam/(len(self.phis)*len(imgs)))
        #if plot==True: self.plotScatter([gbr.flatten(),gbr_ad.flatten()],'Original Gabor filter','Addaptive regulated Gabor filter','Gabor filter after addRegulation')
        return np.array(rmses_lam),np.array(maxOuts_lam),gbr.flatten(),gbr_ad.flatten()

    # Build histoggram
    #   @block: block of real output
    #   @tau: parameters for gaussian weightning
    #   @return   histogram
    def buildHist(self,block,tau=6):
        # gaussian
        def gausian_weightening(x,y):
            return np.exp(-(x**2+y**2)/(2*(tau**2)))/(2*np.pi)

        length =block.shape[0]
        center = length/2
        F_pos=0.0
        F_neg=0.0
        for m in range(0,length):
            for n in range(0,length):
                G = gausian_weightening(m-center,n-center)
                if block[m,n] > 0:
                    F_pos += G*max(0,block[m,n])
                else:
                    F_neg += G*min(0,block[m,n])
        return [F_pos,F_neg]

    # Apply gabor and regularization
    #   @img: image
    #   @return   real part of output
    def applyGabor(self,img,alpha=-1):
        if alpha==-1:alpha=self.alpha
        outputs = []
        for ker in self.kers:
            gbr = cv2.filter2D(img.reshape(self.pixel,self.pixel), cv2.CV_32F,np.real(ker))
            gbr_ad = self.ad_Regulation(np.real(gbr),alpha)
            outputs.append(gbr_ad)
        return np.array(gbr_ad)

    # Extract features
    #   @imgs: buncg of images
    #   @brange: the range of block to calculate histogram
    #   @return   list of features (histograms)
    def extractFeatures(self,imgs,brange,alpha=-1):
        if alpha==-1:alpha=self.alpha
        hist = []
        cc=0
        for img in imgs:
            temp=[]
            rimg = self.applyGabor(img,alpha=alpha)
            # extract features
            for x in range(brange,(self.pixel-brange)):
                for y in range(brange,(self.pixel-brange)):
                    temp.append(self.buildHist(rimg[x-brange:x+brange,y-brange:y+brange],tau=self.tau))
            hist.append(temp)
            cc +=1
            if cc%25==0: print '     done: %d /%d '%(cc,len(imgs))
        hist=np.array(hist)
        return hist.reshape(hist.shape[0],hist.shape[1]*hist.shape[2])

    def plotScatter(self,xy,xlab,ylab,name,tnam='',dir_o='',labels=[]):
        if(len(plt.get_fignums())>0):plt.close()
        if len(xy)==1:plt.scatter(xy[0][0],xy[0][1])
        else:# for multiple dataset
            x = np.arange(len(xy))
            ys = [i+x+(i*x)**2 for i in range(len(xy))]
            colors = cm.jet(np.linspace(0, 1, len(ys)))
            for i in range(0,len(xy)):
                plt.scatter(xy[i][0],xy[i][1],color=colors[i])
            m = cm.ScalarMappable(cmap=cm.jet)
            m.set_array(labels)
            cb = plt.colorbar(m)
            cb.set_label('Alpha')
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        if tnam != '':plt.title(tnam)
        if dir_o == '':dir_o=self.dir_output
        plt.savefig(os.path.join(dir_o,name+'.png'))
        plt.close()

    def plot3D(self,xyz,xlab,ylab,zlab,name,tnam='',dir_o=''):
        if(len(plt.get_fignums())>0):plt.close()
        fig=pylab.figure()
        ax = Axes3D(fig)

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_zlabel(zlab)

        ax.scatter3D(xyz[0][0],xyz[0][1],xyz[0][2],color = (0.0,0.0,1.0),marker="o",s=50,label=u"Test")
        #ax.scatter3D(iris_A[:,0:1],iris_A[:,1:2],iris_A[:,2:3],color = (0.0,0.0,1.0),marker="o",s=50,label=u"Setosa")
        #ax.scatter3D(iris_B[:,0:1],iris_B[:,1:2],iris_B[:,2:3],color = (1.0,0.0,0.0),marker="o",s=50,label=u"Vesicolor")
        #ax.scatter3D(iris_C[:,0:1],iris_C[:,1:2],iris_C[:,2:3],color = (0.0,1.0,0.0),marker="o",s=50,label=u"Verginica")

        if tnam != '':plt.title(tnam)
        if dir_o == '':dir_o=self.dir_output
        fig.savefig(os.path.join(dir_o,name+'.png'))
        plt.close()
        #pylab.show()

    def showGFilter(self,ker):
        #showimg(np.real(ker))
        #showimg(np.imag(ker))
        #ker /= 1.5*ker.sum()   # weighted kernel
        ker =np.real(ker)
        output = cv2.filter2D(self.img_tr[0].reshape(self.pixel,self.pixel), cv2.CV_32F,ker)
        #output = cv2.filter2D(img, cv2.CV_32F,ker)
        showimg(output)
        #output = power(img_f,ker)
        #showimg(output)
        #showimg(cv2.resize(ker, (ksize*20,ksize*20)))
        showimg(ker)



#####################################################################
    def gaborMain(self):
#####################################################################
        # 1. Build several kernels
        if not self.ksize%2: self.ksize += 1
        self.lamda = 2.*self.width

        # estimate range of sigma
        print '    Build several kernel'
        phi_d_max = np.max(self.phis)
        sinval=np.sin(phi_d_max/2.)
        sig1=self.lamda/(sinval*4.*np.sqrt(2)*np.pi)
        sig2=self.width*np.sqrt(2)
        sigmas = np.linspace(min(sig1,sig2),max(sig1,sig2),10)
        #sigmas = np.linspace(0.1,4,100)
        if(DEBUG==1):sigmas = np.linspace(min(sig1,sig2),max(sig1,sig2),5)

        #################################
        # 2. Select sigma using Entropy Correlation Coefﬁcient
        print '    Select sigma'
        tt=time.time()
        imgList = []
        av_eccs = []
        cc = 0
        if self.opt==True:
            nfol=2
            nitr=1
            @optunity.cross_validated(x=self.img_tr, num_folds=nfol, num_iter=nitr)
            def sig_similarity(x_train,x_test,sigma):
                ecc,imgs = self.compareGabor(sigma=sigma,imgs=x_train)
                print '     AECCS of sigma %.3f: %.3f'%(sigma,ecc/len(x_train))
                return ecc/len(x_train)
            kwargs = {'sigma':[min(sigmas),max(sigmas)]}
            optpars, optpars2, optpars3 = optunity.minimize(sig_similarity,**kwargs)
            self.sigma=optpars['sigma']
            self.results.extend([optpars3['solver_name'],str(optpars3['num_generations']),str(optpars3['num_particles']),str(nfol),str(nitr)])
        else:
            for sig in sigmas: # iterate over sigma
                cc += 1
                ecc,imgs = self.compareGabor(sigma=sig,imgs=self.img_tr)
                imgList.extend(imgs)
                av_eccs.append(ecc/len(self.img_tr))
                print '     AECCS of sigma%d: %.3f'%(cc,ecc/len(self.img_tr))

            # sigma which produces the lowest average eccs
            av_eccs = np.delete(av_eccs,np.where(av_eccs==0.))
            idx = np.where(av_eccs==min(av_eccs))[0][0]
            self.showLineGraph(sigmas,np.array(av_eccs),'Sigma','Average ECC')
            self.sigma = sigmas[idx]
            self.results.extend(['','','','',''])
        self.results.append(str(round(time.time()-tt,3)))
        drawimg(np.array(imgList),sigmas,self.dir_output,self.bname+'ker',self.pixel,ncol=20,nrow=6)

        #################################
        # 3. Adaptive regulation of outputs of Gabor filters
        print '    Adaptive regulation'
        tt=time.time()
        alphas = np.linspace(0,5,10)
        if DEBUG ==1:alphas = np.linspace(0,10,10)
        self.lamdas = np.linspace(0.5,self.lamda,self.lamda*2)
        if self.opt==True:
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
            self.results.extend([optpars3['solver_name'],str(optpars3['num_generations']),str(optpars3['num_particles']),str(nfol),str(nitr)])
        else:
            sums_rmse=[]
            gbrs=[]
            for alpha in alphas:
                rmses_lam,maxOuts_lam,gbr,gbr_ad=self.lossFun_alpha(lamdas=self.lamdas,alpha=alpha)
                sums_rmse.append(rmses_lam.sum())
                print '     RMSE of alpha%d: %.3f'%(alpha,rmses_lam.sum())
                # plot
                gbrs.append([gbr,gbr_ad])
                self.showLineGraph(self.lamdas,maxOuts_lam,'Lamdas','Max output with a%d'%alpha )
            sums_rmse = np.array(sums_rmse)
            idx = np.where(sums_rmse==min(sums_rmse))[0][0]
            self.alpha = alphas[idx]
            self.results.extend(['','','','',''])
            #plot
            self.showLineGraph(alphas,sums_rmse,'Alphas','RMSE of max L ')
            self.plotScatter(gbrs,'Original Gabor filter','Addaptive regulated Gabor filter','Gabor filter after addRegulation',labels=alphas)
        self.results.append(str(round(time.time()-tt,3)))
        print '    Adaptive regulation:%.2f'%(time.time()-tt)

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
        self.results.extend([str(self.width),str(self.ksize),str(self.sigma),str(self.lamda),str(len(self.phis)),str(self.alpha),str(self.blocksize),str(self.tau),str(cp)])
        self.csvWriter.writerow(self.results)
        self.f.close()
        del self.csvWriter,self.f

        params = '%s;%s;%s;%s;%s;%s;%s;%s;%s'%(str(self.width),str(self.ksize),str(self.sigma),str(self.lamda),str(len(self.phis)),str(self.alpha),str(self.blocksize),str(self.tau),str(cp))
        return [xtrain, xtest,params]


#####################################################################
    def __init__(self,pixel,img_tr,img_te, dir_input, dir_output,bname,w,opt=False,ksize=21):
#####################################################################
        # set Parameters
        self.dir_input = dir_input
        self.dir_output= dir_output
        self.bname = bname
        self.img_tr=img_tr
        self.img_te=img_te
        self.pixel=pixel
        self.opt=opt

        self.width = w #width = 4. #4. # stroke width
        self.ksize = ksize # kernel size
        #ksize = width
        self.phis = np.array([-np.pi/2,-np.pi/4,0,np.pi/4]) # orientation

        csvname=os.path.join(os.path.dirname(dir_output),'gabourResults.csv')
        self.f = open(csvname, 'ab+')
        self.csvWriter = csv.writer(self.f)
        self.results=[str(dir_output.split('\\')[-1])]






