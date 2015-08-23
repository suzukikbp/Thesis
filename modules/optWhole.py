#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      KSUZUKI
#
# Created:     11/08/2015
# Copyright:   (c) KSUZUKI 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os,cv2,time,csv,copy,pylab
import numpy as np
import optunity,optunity.metrics
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from modules.classification import *
from modules.evaluation import *

from modules.pca import *
from modules.hog import *
from modules.canny import *
from modules.bilateral import *
from modules.ccode import *
from modules.gabor import *
from modules.gabor_cv import *
from modules.gabor_opt1 import *
from modules.gabor_opencv import *



def featureComb(val_comb=None):
    comb=''
    for d in range(1,nb_feature+1):
        comb=comb+ (1 if val_comb/(10*d) < 5 else 0)
    return comb

def extractFeatures(obj,comb,x_train,x_test,**kargs):
    trains,tests=[],[]
    if comb[0]==1: # PCA
        [fdatas.append(dat) for fdatas,dat in zip([trains,tests],[PCA().main([x_train,x_test],m.dir_output,bname)])]
    if comb[1]==1: #Chaincode/Bluring
        datas=[chaincode(dat,m.pixel,m.dir_output)[0]for dat in[x_train,x_test]]
        [fdatas.append(dat) for fdatas,dat in zip([trains,tests],datas)]
    if comb[2]==1: #Bilateral
        kargs_b={'VAL':False,'diameter_bil':kargs['diam'],'sigCol_bil':kargs['sigCol'],'sigSpace_bil':kargs['sigSpace']}
        dat1,dat2,_,_=Bilateral().main(obj.pixel,x_train,x_test,None,\
                    obj.dir_input,obj.dir_output,obj.dataName,**kargs_b)

    return trains,tests

def setModel(algorithm=None, n_neighbors=None, n_estimators=None, max_features=None,
                    kernel=None, C=None, gamma=None, class_weight=None, degree=None, coef0=None):
    if algorithm == 'k-nn':
        model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
    elif algorithm == 'SVM':
        if C:C=10**C
        if gamma:gamma=10**gamma
        model = train_svm(kernel, C, gamma,degree,coef0,class_weight)
    elif algorithm == 'naive-bayes':
        model = GaussianNB()
    elif algorithm == 'random-forest':
        model = RandomForestClassifier(n_estimators=int(n_estimators),max_features=int(max_features))
    else:
        raise ArgumentError('Unknown algorithm: %s' % algorithm)
        return None
    params=[algorithm+'_'+str(kernel),'Optunity_mcl',C,gamma,degree,coef0,class_weight,n_estimators,max_features]
    return model,params

def train_svm(kernel,C, gamma,degree,coef0,w):
    if not w==None:w={0:w}
    if kernel == 'linear':
        model = SVC(kernel=kernel,C=C,class_weight=w)
    elif kernel == 'poly':
        model = SVC(kernel=kernel,C=C,degree=degree,coef0=coef0,class_weight=w)
    elif kernel == 'rbf':
        model = SVC(kernel=kernel,C=C,gamma=gamma,class_weight=w)
    else:
        raise ArgumentError("Unknown kernel function: %s" % kernel)
    return model

def optWhole(self,results,num_folds=2, num_iter=10,num_evals=100):
    tt=time.time()
    @optunity.cross_validated(x=self.xtrain, y=self.ytrain, num_folds=num_folds, num_iter=num_iter)
    def performance(x_train, y_train, x_test, y_test,**kargs):
        # convert to binary expression for features
        comb=featureComb(**kargs)
        # select and calculate all features
        x_train,x_test=extractFeatures(self,comb,x_train,x_test,**kargs)
        model,_=setModel(**kargs)
        model.fit(x_train, y_train)
        # predict the test set
        if kargs['algorithm'] == 'SVM':
            predictions = model.decision_function(x_test)
        else:
            predictions = model.predict_proba(x_test)[:, 1]
        return optunity.metrics.roc_auc(y_test, predictions)

    optimal_configuration, info, optimal_pars3 = optunity.maximize_structured(performance,\
                            search_space=self.search,num_evals=num_evals)

    self.exportResults_pkl(info.call_log,self.bname+'svmrbf'+ ('_weight' if self.CLSW else ''))
    results.extend([max(info.call_log['values']),time.time()-tt])

    return model,results


