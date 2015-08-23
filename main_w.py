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
from modules.evaluation import *
from modules.optWhole import *

DEBUG=1


class FselectClsf(Classification):

    def main(self,dgts,classweight=True,DEBUG=0):
        self.DEBUG=DEBUG
        self.CLSW=classweight
        self.nb_feature=len(self.featureExtrs)

        for i in range(dgts[0],dgts[1]):
            print "\n%s, digit %d "%(self.dataName,i)
            # file setting
            f = open(self.csvname, 'ab+')
            self.csvWriter = csv.writer(f)
            self.dir_output_dgt =os.path.join(self.dir_output,str(i))
            if not os.path.exists(self.dir_output_dgt):os.mkdir(self.dir_output_dgt)
            # initialization
            self.ROCs = {}
            self.basicResults=[self.dataName,self.numEx,self.CLSW]
            # set Target digit
            self.numFeature= 50
            self.setTarget(i)
            self.setParams()
            self.addParams()

            # Classification
            self.bname=self.dataName+'_whole'+('_weight_d' if self.CLSW else '_d')+str(i)+'_'
            model,params,results = optWhole(self,copy.copy(self.basicResults))
            for data in ['test','train']:
                results_,_=self.evaluateModel(model,copy.copy(results),'debug',data=data)
                # export
                results_.append(self.fparams)
                self.csvWriter.writerow(results_)


            # drawing graph
            multipleROC(self.ROCs,self.dir_output_dgt,self.bname)
            del self.csvWriter
            f.close()

    def addParams(self):
        tempdict=self.search
        self.search=dict()
        self.search['part']=tempdict
        fdic={'PCA':None,
            'Chain':None,
            'Bilateral':{'diam':[5,10],'sigCol':[50,200],'sigSpace':[50,200]}}
        self.search['part']['val_comb']=[0,10**(len(self.featureExtrs))]
        self.search['part']['feature']=fdic











if __name__ == '__main__':

    m=FselectClsf()
    m.dataSets = ['mnist_basic','mnist_noise_variation','mnist_background_images','mnist_background_random','mnist_rotation','mnist_rotation_back_image','mnist_rotation_new']
    #m.featureExtrs = ['PCA','ChainBlur','Gabor_set','Bilateral','Canny','Hog']
    m.featureExtrs = ['PCA','ChainBlur','Bilateral']

    if DEBUG==1:
        print 'DEBUG (SMALL SAMPLES, SETTING...ETC)'
        #m.featureExtrs = ['HOG']

    # set Parameters
    m.pixel = 28 # pixel size (only for a square)
    noiselevel=6
    m.dir_input = "..\data\input"
    m.dir_output = "..\data\output"
    m.dir_data = "..\data\output\data"
    m.dataName =m.dataSets[0]
    m.numEx = int(time.time())
    m.trainSize=m.valSize=m.teSize=200 #50000
    m.normalization = False # MNIST is already normalized
    if DEBUG==1:m.trainSize=m.valSize=m.teSize=100#60

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


    m.xtrain,m.xtest,m.xval=img_tr_all,img_te_all,img_vl_all

    # Extract features & classification
    digits=[0,10]        # digits number for classification
    if DEBUG ==1:
        digits=[0,1]

    for classweight in[False,True]:
        mm=copy.copy(m)
        m.main(digits,classweight=classweight,DEBUG=DEBUG)


    print "finish"



