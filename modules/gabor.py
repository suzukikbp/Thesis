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


class Gabor():
    def __init__(self,pixel,img_tr,img_te,img_vl,dir_in,dir_out,bname,**kargs):
        self.SHOW = 0
        self.DEBUG = 0
        self.EXP = True
        if self.DEBUG==1:print 'DEBUG MODE'

        # 1. Set initial Parameters
        self.dir_input,self.dir_output,self.bname = dir_in,dir_out,bname
        self.img_tr,self.img_te,self.img_vl=img_tr,img_te,img_vl
        self.pixel=pixel
        self.results=[str(dir_out.split('\\')[-1])]
        csvname=os.path.join(os.path.dirname(dir_out),'gabourResults.csv')
        self.f = open(csvname, 'ab+')
        self.csvWriter = csv.writer(self.f)

        if kargs['opt']:self.optParams(**kargs)
        else:self.setParams(**kargs)

    # set parameters
    def setParams(self,**kargs):
        self.ksize=kargs['ks']
        self.sigma=kargs['sigma']
        self.lamda=kargs['lmd']
        self.N=kargs['n']
        self.D=kargs['d']
        self.nbPhi=kargs['nbPhi']
        self.alpha=kargs['alpha']
        self.width=-1
        self.phis=np.linspace(-np.pi/2,np.pi/4,self.nbPhi)
        print '    === Optimal patams ===\n    w:%d, #phis:%d, ksize:%d, D:%d(N:%d),sig:%.3f,alpa:%.2f'%(self.width,self.nbPhi,self.ksize,self.D,self.N,self.sigma,self.alpha)

    # optimize parameters
    def optParams(self,**kargs):
        params1=[]
        params2=[]
        self.check=[]
        for w in kargs['ws']:
            self.linex=[]
            self.liney=[]
            self.labs=[]
            for nbPhi in kargs['nbPhis']:
                for n in kargs['ns']:
                    tmp=self.setSigma(w,nbPhi,n)
                    params1.append(tmp[0:2])
                    params2.append(tmp[2:4])
            self.plotMultiLines(self.linex,self.liney,xlab='Sigma',ylab='Average ECC',tnam='Width %d'%w,multi=True,labels=self.labs)

        # Set optimal parameters
        params=np.array(params1).transpose()
        self.eccs=min(params[0])
        idx = np.where(params[0]==self.eccs)[0][0]
        self.sigma=params[1][idx]
        ps=params2[idx]
        self.results.extend(ps[0])
        self.width = ps[1][0]
        self.N = ps[1][4]
        self.nbPhi=ps[1][1]
        self.phis = np.linspace(-np.pi/2,np.pi/4,self.nbPhi)
        self.lamda = 2.*self.width
        self.ksize = ps[1][2]
        self.D = ps[1][3]

        # Additional regulation
        self.setAlpha()


    # N:the number of block in the images
    # w:width of stroke = kernel size
    #####################################################################
    def setSigma(self,w,nbPhi,N):
    #####################################################################
        # 1. Set params(width,phis,lamda,ksize,S,N) under the constraints
        width = w # width of stroke = kernel size
        phis = np.linspace(-np.pi/2,np.pi/4,nbPhi) # orientation of the stroke
        lamda = 2.*width  # wavelength of simusdidal factor (scale)
        ksize = width     # kernel size (=M)
        D = self.pixel/N  # sampling interval (# pixel)

        ksize,N,D,nbPhi=self.modifyParam(ksize,N,nbPhi)
        fn ='W%sL%sNP%sK%sD%s'%(str(w).zfill(2),str(lamda).zfill(2),str(nbPhi).zfill(2),str(ksize).zfill(2),str(D).zfill(2))

        # check if above params are already tried or not
        if fn in self.check:return [np.inf,np.inf,np.inf,np.inf]
        else:self.check.append(fn)

        # 2. Calculate the range of sigmas
        sigmas =self.estSigRange(phis,lamda,width)# range of sigma
        print '    w:%d, lmd:%d, #phis:%d, ksize:%d, D:%d(N:%d)'%(w,lamda,nbPhi,ksize,D,N)

        #################################
        # 2. Select sigma using Entropy Correlation Coefﬁcient
        print '    Select sigma'
        tt=time.time()
        imgList = []
        av_eccs = []
        self.exportImgs=[]
        cc = 0
        for sig in sigmas: # iterate over sigma
            cc += 1
            ecc,imgs = self.compareGabor(sigma=sig,ksize=ksize,lamda=lamda,phis=phis,N=N,imgs=self.img_tr)
            imgList.extend(imgs)
            av_eccs.append(ecc/len(self.img_tr))
            print '     AECCS of sigma%d: %.3f'%(cc,ecc/len(self.img_tr))

        # sigma which produces the lowest average eccs
        av_eccs = np.delete(av_eccs,np.where(av_eccs==0.))
        fin_eccs=min(av_eccs)
        idx = np.where(av_eccs==fin_eccs)[0][0]
        self.showLineGraph(sigmas,np.array(av_eccs),'Sigma','Average ECC',tnam=fn)
        self.linex.append(sigmas)
        self.liney.append(np.array(av_eccs))
        self.labs.append('D: %d, nbPhi: %d'%(D,nbPhi))
        fin_sigma = sigmas[idx]
        params=['','','','','']

        params.append(str(round(time.time()-tt,3)))
        if self.EXP:
            lab = ['' for i in range(len(self.exportImgs))]
            ncol=nbPhi*2+1#ncol=int(scm.comb(nbPhi,2))*2+1
            for i in range(0,len(sigmas)):
                lab[i*ncol]='sigma=%.2f'%sigmas[i]
            drawimg(self.exportImgs,lab,self.dir_output,self.bname+'_ker_'+fn,self.pixel,ncol=ncol,nrow=len(sigmas),skip=False,tnam='Width: %d, ksize: %d, #phi:%d, N: %d (D: %d)'%(w,ksize,nbPhi,N,D))
        return [fin_eccs,fin_sigma,params,[w,nbPhi,ksize,D,N]]


    # Adaptive regulation of outputs of Gabor filters
    #####################################################################
    def setAlpha(self):
    #####################################################################
        print '    Adaptive regulation'
        tt=time.time()
        alphas = np.linspace(1,5,10)#(0.5,5,10)
        if self.DEBUG ==1:alphas = np.linspace(1,10,3)#(0.5,10,3)
        self.lamdas = np.linspace(0.001,self.lamda,10)
        sums_rmse,gbs,gbas,gbrs=[],[],[],[]
        for alpha in alphas:
            rmses_lam,maxOuts_lam,gbr,gbr_ad=self.lossFun_alpha(lamdas=self.lamdas,alpha=alpha)
            sums_rmse.append(rmses_lam.sum())
            print '     RMSE of alpha%d: %.3f'%(alpha,rmses_lam.sum())
            # plot
            gbs.append(gbr)
            gbas.append(gbr_ad)
            gbrs.append([gbr,gbr_ad])
            self.showLineGraph(self.lamdas,maxOuts_lam,'Lamdas','Max output with a%d'%alpha )
        sums_rmse = np.array(sums_rmse)
        idx = np.where(sums_rmse==min(sums_rmse))[0][0]
        self.alpha = alphas[idx]
        self.results.extend(['','','','',''])
        #plot
        self.showLineGraph(alphas,sums_rmse,'Alphas','RMSE of max L ')
        lab = ['' for i in range(0,len(gbs+gbas))]
        for i in range(0,len(alphas)):
            lab[i]='alpha:%.1f'%alphas[i]
        drawimg(gbs+gbas,lab,self.dir_output,self.bname+'_alpha_',self.pixel,ncol=len(alphas),nrow=2,skip=False)
        self.plotScatter(gbrs,'Original Gabor filter','Addaptive regulated Gabor filter','Gabor filter after addRegulation',labels=alphas)
        self.results.append(str(round(time.time()-tt,3)))
        print '    Adaptive regulation:%.2f'%(time.time()-tt)


    #####################################################################
    def applyGabor(self):
    #####################################################################

        # 1. Set Gabor kernel with optimal params
        self.kers = [] # kernels (each phi)
        for phi in self.phis:
            self.kers.append(self.buildKernel(self.ksize,self.sigma, phi ,self.lamda))
        self.kers=np.array(self.kers)
        print '    === Optimal patams ===\n    w:%d, #phis:%d, ksize:%d, D:%d(N:%d),sig:%.3f,alpa:%.2f'%(self.width,self.nbPhi,self.ksize,self.D,self.N,self.sigma,self.alpha)

        #################################
        # 2. Produce output image and extract features: local histgram
        self.blocksize = 5
        self.tau = 6
        brange =self.blocksize/2
        print '    Feature selection train/test/val'
        hist_tr,hist_te,hist_vl = [self.extractFeatures(dat,brange)for dat in [self.img_tr,self.img_te,self.img_vl]]

        #################################
        # 3. Compress dimentionality : PCA
        xtrain,xtest,xval,cp = PCA().main([hist_tr,hist_te,hist_vl],self.dir_output,self.bname)
        print '    PCA: %d'%cp

        #################################
        # 4. Export results and return
        self.results.extend([str(self.width),str(self.ksize),str(self.sigma),str(self.lamda),str(len(self.phis)),str(self.alpha),str(self.blocksize),str(self.tau),str(cp),str(self.eccs)])
        self.csvWriter.writerow(self.results)
        self.f.close()
        del self.csvWriter,self.f

        params = '%s;%s;%s;%s;%s;%s;%s;%s;%s'%(str(self.width),str(self.ksize),str(self.sigma),str(self.lamda),str(len(self.phis)),str(self.alpha),str(self.blocksize),str(self.tau),str(cp))
        return [xtrain, xtest,xval,params]


#####################################################################
#### Functions ######################################################
#####################################################################

    #  Apply constraints to params
    def modifyParam(self,ksize,n,nbPhi):

        # modify the number of phi
        if nbPhi<2:nbPhi=2

        # modify kernel size
        ksize=min(int(ksize),int(self.pixel-2))
        # make ksize to odd number
        if not ksize%2: ksize = ksize+1

        # modify N
        n=int(n)
        if n==0:n=1
        elif n>self.pixel:n=self.pixel

        # modify D sampling interval (# pixel)
        D = self.pixel/n
        if ksize < D: # D < ksize
            D =ksize
        try:
            while self.pixel%D!=0:D-=1
        except:
            print 'D value error D %d, ksize %d, N %d '%(D,ksize,n)

        # modify N according to D
        n =self.pixel/D
        return ksize,n,D,nbPhi

    #  Estimate range of sigma
    def estSigRange(self,phis,lamda,width):
        phi_d_max = np.max(phis)
        sinval=np.sin(phi_d_max/2.)
        sig1=lamda/(sinval*4.*np.sqrt(2)*np.pi)
        sig2=width*np.sqrt(2)
        sigmas = np.linspace(min(sig1,sig2),max(sig1,sig2),10)
        #sigmas = np.linspace(0.1,4,100)
        if(self.DEBUG==1):sigmas = np.linspace(min(sig1,sig2),max(sig1,sig2),3)
        return sigmas

    def buildKernel(self,ks, sigma, phi ,lamda):
        # interval
        #xs=ys=np.linspace(-ks/2.,ks/2.,ks)
        xs=ys=np.linspace(-1.,1.,ks)
        x,y = np.meshgrid(xs,ys)
        # rotation
        R1 = x*np.cos(phi)+y*np.sin(phi)
        R2 =-x*np.sin(phi)+y*np.cos(phi)

        exy = -0.5/(sigma**2)
        cscale = (2.*np.pi)/lamda

        scale=1./(2*np.pi*(sigma**2))
        b =np.exp(exy*(R1**2+R2**2))
        c =np.exp(1j*(cscale*R1))
        ker = np.array(scale*b*c,dtype=np.complex128)

        ## algorithm of opencv
        #scale = 1.
        #b =np.exp(exy*(R1**2+R2**2))
        #c =np.cos(cscale*R1)
        #ker = np.array(scale*b*c,dtype=np.float32)

        #return np.array(np.exp(-0.5*(R1**2+R2**2)/sigma**2)*np.cos(2.*np.pi*R1/lamda),dtype=np.float32)

        return ker

    def power(self,img, ker):
        return np.sqrt(ndi.convolve(img, np.real(ker), mode='wrap')**2 +
                       ndi.convolve(img, np.imag(ker), mode='wrap')**2)

    # Convolution
    #   @img: input image
    #   @ker: kernel
    #   @return output: output image
    def convol(self,img,ker,ksize=-1,N=-1):
        if ksize==-1:ksize=self.ksize
        if N==-1:N=self.N

        M = ksize    #kernel size (#pixel)
        D = self.pixel/N  #sampling interval (#pixel)
        hM = (M-1)/2        # a half of M
        hD = D/2            # a half of D
        dif = int(M-D)      # teh difference between kernel and block
        tmpimg,tmpcnt=np.zeros([self.pixel,self.pixel]),np.zeros([self.pixel,self.pixel])

        for ox in np.linspace(0,self.pixel-D,N):# iterate over blocks, origin is upper left cell
            for oy in np.linspace(0,self.pixel-D,N):
                # cordination of kernel
                sx,ex=ox-hM,ox+hM
                sy,ey=oy-hM,oy+hM
                k=ker.copy()
                if sx<0: k=k[-int(sx):,:]
                if sy<0: k=k[:,-int(sy):]
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    try:
                        if self.pixel-1<ex:k=k[:-int(ex-(self.pixel-1)),:]
                        if self.pixel-1<ey:k=k[:,:-int(ey-(self.pixel-1))]
                        #if self.pixel<ex:k=k[:-int(ex-self.pixel),:]
                        #if self.pixel<ey:k=k[:,:-int(ey-self.pixel)]
                    except Warning:
                        print 'warning'

                block = img[max(0,sx):min(self.pixel-1,ex+1),max(0,sy):min(self.pixel-1,ey+1)]
                try:
                    tmpimg[max(0,sx):min(self.pixel-1,ex+1),max(0,sy):min(self.pixel-1,ey+1)] \
                        += cv2.filter2D(block, cv2.CV_32F,np.real(k))
                    tmpcnt[max(0,sx):min(self.pixel-1,ex+1),max(0,sy):min(self.pixel-1,ey+1)] \
                        += 1
                except:
                    print "error_conv_plus"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.array(tmpimg*1.0/tmpcnt)
        warnings.resetwarnings()
        out[np.isnan(out)]= 0
        return out

    # Compare output of Gabor filter with different phis
    #   @sigma: to build kernel
    #   @imgs: images to apply gabor filter
    #   @return ecc: entropy correlation coefficient
    #   @return imgs: output applied gabor filter
    def compareGabor(self,ksize=-1,lamda=-1,sigma=0,phis=[],N=-1,imgs=[]):
        if len(imgs)==0:imgs=self.img_tr
        if ksize==-1:ksize=self.ksize
        if lamda==-1:lamda=self.lamda
        if len(phis)==0:phis=self.phis
        if N==-1:N=self.N

        ecc = 0.
        nbimg=0
        imgList=[]
        for img in imgs:
            nbimg += 1
            ecc_img= 0.
            c=0
            temp = []
            for i in range(0,len(phis)): # iterate over phis
                for j in range(0,len(phis)):
                    if i>j:
                        c +=1
                        #print '     sig: %0.2f, phi:%0.2f'%(sig,phi)
                        keri =self.buildKernel(ksize, sigma, phis[i] ,lamda)
                        kerj =self.buildKernel(ksize, sigma, phis[j] ,lamda)
                        if self.SHOW == 1:self.showGFilter(keri)
                        # compare output of Gabor filter with different phis
                        gbr_i = self.convol(img.reshape(self.pixel,self.pixel),np.real(keri),ksize=ksize,N=N)
                        gbr_j = self.convol(img.reshape(self.pixel,self.pixel),np.real(kerj),ksize=ksize,N=N)
                        #gbr_i = cv2.filter2D(img.reshape(self.pixel,self.pixel), cv2.CV_32F,np.real(keri))
                        #gbr_j = cv2.filter2D(img.reshape(self.pixel,self.pixel), cv2.CV_32F,np.real(kerj))

                        if self.EXP:
                            if nbimg==1:
                                if c==1:self.exportImgs.append(img)
                                if not j in temp:
                                    self.exportImgs.extend([gbr_j,np.real(kerj)])
                                    temp.append(j)
                                if not i in temp :
                                    self.exportImgs.extend([gbr_i,np.real(keri)])
                                    temp.append(i)
                            #showimg(gbr_i)
                        ecc_img += self.calcECC(gbr_i,gbr_j)
                        if np.isnan(gbr_i.flatten()[0]):
                            print 'error'
                        imgList.append(gbr_i)
            ecc += ecc_img/c
        return ecc,imgList

    # Calculate Entropy Correlation Coefﬁcient (ECC)
    def calcECC(self,img_i,img_j):

        # Normalization by Scaling Between 0 and 1
        def normalization(img,min_,max_):
            if max_-min_==0:
                print 'error_normalization'
            return np.array(((img-min_)/(max_-min_))*255,np.uint8)
        minVal = min(img_i.min(),img_j.min())
        maxVal = max(img_i.max(),img_j.max())
        img_i = normalization(img_i,minVal,maxVal)
        img_j = normalization(img_j,minVal,maxVal)

        # histgoram of images
        hist_i = np.bincount(img_i.ravel(),minlength=256)
        hist_j = np.bincount(img_j.ravel(),minlength=256)

        if(self.SHOW==1):
            plt.subplot(1,2,1);plt.plot(hist_i);
            plt.subplot(1,2,2);plt.plot(hist_j);
            plt.show()
        # calculate entropy
        hi,hj,hij,temp = 0.0,0.0,0.0,0.0
        sum_i,sum_j = np.sum(hist_i),np.sum(hist_j)
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
                    gbr = self.convol(img.reshape(self.pixel,self.pixel),np.real(ker))
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

    # Set gabor and regularization
    #   @img: image
    #   @return   real part of output
    def setGabor(self,img,alpha=-1):
        if alpha==-1:alpha=self.alpha
        outputs = []
        for ker in self.kers:
            gbr = self.convol(img.reshape(self.pixel,self.pixel),np.real(ker))
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
            rimg = self.setGabor(img,alpha=alpha)
            # extract features
            for x in range(brange,(self.pixel-brange)):
                for y in range(brange,(self.pixel-brange)):
                    temp.append(self.buildHist(rimg[x-brange:x+brange,y-brange:y+brange],tau=self.tau))
            hist.append(temp)
            cc +=1
            if cc%25==0: print '     done: %d /%d '%(cc,len(imgs))
        hist=np.array(hist)
        return hist.reshape(hist.shape[0],hist.shape[1]*hist.shape[2])

    #############################
    ### description functions ###
    #############################
    def showLineGraph(self,data1,data2,xlab,ylab,tnam=''):
        plt.clf()
        plt.plot(data1,data2)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        if tnam != '':plt.title(tnam)
        #plt.legend()
        plt.savefig(os.path.join(self.dir_output,self.bname+'_'+ylab+'_'+tnam+'.png'))


    def plotMultiLines(self,x,y,xlab='',ylab='',tnam='',multi=False,labels=''):
        if(len(plt.get_fignums())>0):plt.close()
        plt.figure()
        if multi:
            for i in range(0,len(x)):
                plt.plot(x[i], y[i], label=labels[i])
        else:
            plt.plot(x, y, label=labels)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(tnam)
        plt.legend(loc="lower right")
        #plt.show()
        #plt.savefig((name+".png"), dpi=100)
        plt.savefig(os.path.join(self.dir_output,self.bname+'_'+ylab+'_'+tnam+'.png'))
        plt.close()


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

        if tnam != '':plt.title(tnam)
        if dir_o == '':dir_o=self.dir_output
        fig.savefig(os.path.join(dir_o,name+'.png'))
        plt.close()
        #pylab.show()

    def showGFilter(self,ker,cvshow=True):
        #showimg(np.real(ker))
        #showimg(np.imag(ker))
        #ker /= 1.5*ker.sum()   # weighted kernel
        ker =np.real(ker)
        if cvshow: output=cv2.filter2D(self.img_tr[0].reshape(self.pixel,self.pixel), cv2.CV_32F,ker)
        else: output=self.convol(self.img_tr[0].reshape(self.pixel,self.pixel),np.real(ker))
        #output = cv2.filter2D(img, cv2.CV_32F,ker)
        showimg(output)
        #output = power(img_f,ker)
        #showimg(output)
        #showimg(cv2.resize(ker, (ksize*20,ksize*20)))
        showimg(ker)











