# encoding: utf-8

#-------------------------------------------------------------------------------
# Name:        main
# Purpose:
#
# Author:      KSUZUKI
#
# Created:     30/01/2015
# Copyright:   (c) KSUZUKI 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import os,cv2,time,csv,copy,pylab
import numpy as np
import optunity,optunity.metrics
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

from modules.preposs import *
from modules.classification import *
from modules.pca import *
from modules.hog import *
from modules.canny import *
from modules.bilateral import *
from modules.ccode import *
from modules.evaluation import *
from modules.gabor import *
from modules.gabor_cv import *
from modules.gabor_opt1 import *
from modules.gabor_opencv import *

SHOW=1
DEBUG=0
CLS=1

dataSets = ['mnist_basic','mnist_noise_variation','mnist_background_images','mnist_background_random','mnist_rotation','mnist_rotation_back_image','mnist_rotation_new']
#featureExtrs = ['PCA','ChainBlur','Gabor','Gabor_opt1']
#featureExtrs = ['Gabor_set','PCA','ChainBlur','Gabor_opt1','Gabor']
#featureExtrs = ['HOG','Gabor_set','PCA','ChainBlur']
#featureExtrs = ['HOG']
#featureExtrs = ['ChainBlur','Gabor_opt1','Gabor']
#featureExtrs = ['Gabor_opencv','Gabor_set','Gabor_opt1']
featureExtrs = ['PCA','ChainBlur','Gabor_set','Bilateral','Canny','HOG','Gabor_opencv']
solvers = ['particle swarm','grid search','random search','cma-es','nelder-mead']

#featureExtrs = ['Gabor_opt1']

if DEBUG==1:
    print 'DEBUG (SMALL SAMPLES, SETTING...ETC)'
    featureExtrs = ['Bilateral']
    #featureExtrs = ['Gabor_opt1']
    #featureExtrs = ['ChainBlur']
    #featureExtrs = ['Gabor']
    #opts = ['Default','Gsearch']


if __name__ == '__main__':

    m=Classification()
    # set Parameters
    m.pixel = 28 # pixel size (only for a square)
    noiselevel=1
    m.dir_input = "..\data\input"
    m.dir_output = "..\data\output"
    m.dir_data = "..\data\output\data"
    m.dataName =dataSets[1]
    m.numEx = int(time.time())
    m.trainSize=m.valSize=m.teSize=200 #50000
    m.normalization = False # MNIST is already normalized
    if CLS==DEBUG==1:m.trainSize=m.valSize=m.teSize=100#60
    elif DEBUG==1 and DEBUG==1:m.trainSize=m.valSize=m.teSize=5#60

    print 'No. %d starts'%m.numEx
    #################################
    # 1. Data Load
    #obtain(m.dir_input,m.dataName)

    # output file setting
    if 'noise' in m.dataName:m.dataName=m.dataName+'s_all_'+str(noiselevel)
    m.csvname=os.path.join(m.dir_output,'results_'+m.dataName+'.csv')
    m.dir_output =os.path.join(m.dir_output,str(m.numEx))
    os.mkdir(m.dir_output)
    #m.dataName = '_'.join(m.dataSet.split('_')[1:])

    # set variables
    img_tr_all,m.lab_tr_all,img_vl_all,m.lab_vl_all,img_te_all,m.lab_te_all \
        = load(m.dir_input,m.dataName,m.pixel,trlen=m.trainSize,telen=m.teSize,vlen=m.valSize)
    if m.normalization: img_tr_all,img_te_all,img_te_all = [norm(indat) for indat in [img_tr_all,img_te_all,img_te_all]]
    if not DEBUG:(img_tr_all,m.lab_tr_all,m.dir_output,m.dataName,m.pixel)
    #################################
    # 2. Extract features & classification
    for featureExtr in featureExtrs:
        t = time.time()
        print '\n'+featureExtr
        bname = m.dataName+'_'+featureExtr+'_'

        digits=[0,10]        # digits number for classification
        ws_gb=[3,4,5,7]     # candidates of width
        lms_gb=[1,1000]     # lambda range (just for opt1, will be devided by 100)
        nbPhis_gb=[2,4,6,8] # candidates/ range of the the value (should be more than 1)
        ks_gb=[3,5,7,13]    # candidates / range of kernel size
        ns_gb=[2,4,7,14]    # candidates / range of # block in the images
        sigs_gb=[1,50]      # range of sigma(just for opt1, will be devided by 10)
        numgs_gb=[10]#[3,5,10]      # the number of generalization for PSO
        numps_gb=[10]#[5,10]     # the number of  for PSO
        nfols_gb=[2]#[2,4]      # the number of fold for PSO
        nitrs_gb=[100]#[2,5,10]      # the number of iteration for PSO

        if DEBUG ==1:
            digits=[0,1]
            ks_gb=[3]#[3,9,13]
            ws_gb=[5]#[3,5]
            nbPhis_gb=[6]#,2]#[4,6]
            ns_gb=[14]#[4,7,14,28]
            numgs_gb=[1]
            numps_gb=[1]
            nfols_gb=[2]
            nitrs_gb=[1]

        optparms=m.createCombs(numgs_gb,numps_gb,nfols_gb,nitrs_gb)
        m.fparams=[]

        def classification(m,featureExtr,tm):
            for classweight in[False,True]:
                mm=copy.copy(m)
                mm.main(digits,featureExtr,tm,'optSVM',classweight=classweight,DEBUG=DEBUG)
                mm=copy.copy(m)
                mm.main(digits,featureExtr,tm,'optClfs',classweight=classweight,DEBUG=DEBUG)

        # 1 PCA
        if(featureExtr=='PCA'):
            m.xtrain,m.xtest,m.xval,m.fparams = PCA().main([img_tr_all,img_te_all,img_vl_all],m.dir_output,bname)
            print "%s: %0.2f sec, CP:%d" %(featureExtr,(time.time()-t),m.fparams)
            if CLS==1:classification(m,featureExtr,time.time()-t)
            [m.exportResults_pkl(dat,name,odir=m.dir_data) for dat,name in zip([m.xtrain,m.xtest,m.xval],m.makeFileName(bname,m.fparams))]

        # 2. Chaincode/Bluring
        # directional decomposition of image into directional sub-images
        elif(featureExtr=='ChainBlur'):
            m.xtrain,m.xtest,m.xval=[chaincode(dat,m.pixel,m.dir_output)[0]for dat in[img_tr_all.copy(),img_te_all.copy(),img_vl_all.copy()]]
            print "%s: %0.2f sec" %(featureExtr,(time.time()-t))
            if CLS==1:classification(m,featureExtr,time.time()-t)
            [m.exportResults_pkl(dat,name,odir=m.dir_data) for dat,name in zip([m.xtrain,m.xtest,m.xval],m.makeFileName(bname,m.fparams))]

        # 3.1 Gabor with static way
        elif(featureExtr=='Gabor'):
            kargs={'DEBUG':DEBUG,'opt':True,'ws':ws_gb,'nbPhis':nbPhis_gb,'ns':ns_gb}
            gab=Gabor(m.pixel,img_tr_all,img_te_all,img_vl_all,\
                        m.dir_input,m.dir_output,m.dir_data,m.dataName,**kargs)
            m.xtrain,m.xtest,m.xval,m.fparams=gab.applyGabor(opt=True)
            #if 'Gabor_cv' in featureExtrs:gab_cv= Gabor_cv(gab)
            print "%s: %0.2f sec" %(featureExtr,(time.time()-t))
            if CLS==1:classification(m,featureExtr,time.time()-t)
            [m.exportResults_pkl(dat,name,odir=m.dir_data) for dat,name in zip([m.xtrain,m.xtest,m.xval],m.makeFileName(bname,m.fparams))]

        # 3.2 Gabor with hyperparameter optimization
        elif(featureExtr=='Gabor_opt1'):
            for i in range(0,len(optparms)):
                t=time.time()
                gab= Gabor_opt1(m.pixel,img_tr_all,img_te_all,img_vl_all,\
                            m.dir_input,m.dir_output,m.dir_data,m.dataName,\
                            ws=ws_gb,lms=lms_gb,nbPhis=nbPhis_gb,sigmas=sigs_gb,\
                            ksizes=ks_gb,ns=ns_gb,optparms=optparms[i],DEBUG=DEBUG)
                m.xtrain,m.xtest,m.xval,m.fparams=gab.applyGabor(opt=True)
                print "%s: %0.2f sec" %(featureExtr,(time.time()-t))
                if CLS==1:classification(m,featureExtr,time.time()-t)
                [m.exportResults_pkl(dat,name,odir=m.dir_data) for dat,name in zip([m.xtrain,m.xtest,m.xval],m.makeFileName(bname,m.fparams.split(';')))]
                t=time.time()

        # 3.3 Gabor with static way
        elif(featureExtr=='Gabor_set'):
            args=[]
            args.append({'DEBUG':DEBUG,'opt':False,'ks':13,'sigma':4.645173,'lmd':0.010689,'n':14,'d':2,'nbPhi':4,'alpha':2.312176579,'eccs':0.898335796})
            args.append({'DEBUG':DEBUG,'opt':False,'ks':3,'sigma':4.242640687,'lmd':6.,'n':14,'d':2,'nbPhi':2,'alpha':1.,'eccs':0.955743598})
            for i,kargs in enumerate(args):
                gab=Gabor(m.pixel,img_tr_all,img_te_all,img_vl_all,\
                            m.dir_input,m.dir_output,m.dir_data,m.dataName,**kargs)
                m.xtrain,m.xtest,m.xval,m.fparams=gab.applyGabor(opt=False)
                if CLS==1:classification(m,featureExtr+'_'+str(i),time.time()-t)
                [m.exportResults_pkl(dat,name,odir=m.dir_data) for dat,name in zip([m.xtrain,m.xtest,m.xval],m.makeFileName(bname,m.fparams.split(';')))]
                t=time.time()

        # 3.4 Gabor with opencv
        elif(featureExtr=='Gabor_opencv'):
            kargs={'DEBUG':DEBUG,'opt':True,'ws':ws_gb,'nbPhis':nbPhis_gb,'ns':ns_gb}
            gab=Gabor_opencv(m.pixel,img_tr_all,img_te_all,img_vl_all,\
                        m.dir_input,m.dir_output,m.dir_data,m.dataName,**kargs)
            m.xtrain,m.xtest,m.xval,m.fparams=gab.applyGabor(opt=True)
            #if 'Gabor_cv' in featureExtrs:gab_cv= Gabor_cv(gab)
            print "%s: %0.2f sec" %(featureExtr,(time.time()-t))
            if CLS==1:classification(m,featureExtr,time.time()-t)
            [m.exportResults_pkl(dat,name,odir=m.dir_data) for dat,name in zip([m.xtrain,m.xtest,m.xval],m.makeFileName(bname,m.fparams.split(';')))]

        # 4 HOG
        elif(featureExtr=='HOG'):
            for pix in range(5,m.pixel/3):
                block=m.pixel/pix
                for cel in range(1,block):
                    dimention=8*(cel*cel)*(block-cel+1)*(block-cel+1)
                    if dimention < 200:
                        kargs={'DEBUG':DEBUG,'VAL':False,'pixs':pix,'cells':cel,'orientation':8,'dimention':dimention}
                        m.xtrain,m.xtest,m.xval,m.fparams=Hog().main(m.pixel,img_tr_all,img_te_all,img_vl_all,\
                                    m.dir_input,m.dir_output,m.dataName,**kargs)

                        print "%s: %0.2f sec" %(featureExtr,(time.time()-t))
                        if CLS==1:classification(m,featureExtr,time.time()-t)
                        [m.exportResults_pkl(dat,name,odir=m.dir_data) for dat,name in zip([m.xtrain,m.xtest,m.xval],m.makeFileName(bname,m.fparams))]

        # 5 Bilateral
        elif(featureExtr=='Bilateral'):
            for diam in range(5,6):
                for sigCol in range(200,201):
                    for sigSpace in range(50,51):
                        kargs={'VAL':False,'diameter_bil':diam,'sigCol_bil':sigCol,'sigSpace_bil':sigSpace}
                        m.xtrain,m.xtest,m.xval,m.fparams=Bilateral().main(m.pixel,img_tr_all,img_te_all,img_vl_all,\
                                    m.dir_input,m.dir_output,m.dataName,**kargs)

                        print "%s: %0.2f sec" %(featureExtr,(time.time()-t))
                        if CLS==1:classification(m,featureExtr,time.time()-t)
                        [m.exportResults_pkl(dat,name,odir=m.dir_data) for dat,name in zip([m.xtrain,m.xtest,m.xval],m.makeFileName(bname,m.fparams))]

        # 6 Canny
        elif(featureExtr=='Canny'):
            for diam in range(5,6):
                for sigCol in range(200,201):
                    for sigSpace in range(50,51):
                        for th1 in range(90,91):
                            for th2 in range(90,91):
                                for ksize in range(3,4):
                                    if ksize%2==0:ksize=ksize+1
                                    kargs={'DEBUG':DEBUG,'VAL':False,'diameter_bil':diam,'sigCol_bil':sigCol,'sigSpace_bil':sigSpace,'th1_canny':th1,'th2_canny':th2,'ksize_canny':ksize}
                                    m.xtrain,m.xtest,m.xval,m.fparams=Canny().main(m.pixel,img_tr_all,img_te_all,img_vl_all,\
                                                m.dir_input,m.dir_output,m.dataName,**kargs)

                                    print "%s: %0.2f sec" %(featureExtr,(time.time()-t))
                                    if CLS==1:classification(m,featureExtr,time.time()-t)
                                    [m.exportResults_pkl(dat,name,odir=m.dir_data) for dat,name in zip([m.xtrain,m.xtest,m.xval],m.makeFileName(bname,m.fparams))]
    # concatinate datas
    # select data from folder
    # add data to list (xtrain,x_test)
    # pca?
    # classification

    if CLS==1:
        m.xtrain,m.xtest,m.xval = np.zeros(m.trainSize),np.zeros(m.teSize),np.zeros(m.valSize)
        c=0
        #for dpath,dnames,fnames in os.walk(m.dir_data):
        for fname in os.listdir(m.dir_data):
            if m.dataName in fname:
                if 'train' in fname:
                    if int(fname.split('_')[-1])==m.trainSize:
                        c+=1
                        print fname
                        m.xtrain=np.c_[m.xtrain,m.importResults_pkl(fname,m.dir_data)]
                if 'test' in fname:
                    if int(fname.split('_')[-1])==m.teSize:
                        print fname
                        m.xtest=np.c_[m.xtest,m.importResults_pkl(fname,m.dir_data)]
                if 'val' in fname:
                    if int(fname.split('_')[-1])==m.valSize:
                        print fname
                        m.xval=np.c_[m.xval,m.importResults_pkl(fname,m.dir_data)]
        m.xtrain,m.xtest,m.xval=m.xtrain[:,1:],m.xtest[:,1:],m.xval[:,1:]
        featureExtr='All_f'+str(c)
        bname = m.dataName+'_'+featureExtr+'_'
        # Compress dimentionality : PCA
        m.xtrain,m.xtest,cp = PCA().main([m.xtrain,m.xtest],m.dir_output,bname)
        classification(m,featureExtr,time.time()-t)


    print "finish"
