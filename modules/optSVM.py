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


def optSVM(self,param,results,num_folds=2, num_iter=10,num_evals=100):
    print '\n   '+param
    tt,t=0,time.time()

    results.extend(['SVM_rbf',param])
    if param == 'Default':
        num_folds=num_iter=num_evals=num_gen=num_par=optima=-1
        model =svm.SVC(kernel='rbf',probability=True)
        results.extend([model.C,model.gamma,'','',None])

    elif param == 'Gsearch':
        num_iter=num_evals=num_gen=num_par=optima=-1
        args = {'kernel': ['rbf'], 'C':self.crange, 'gamma':self.grange}
        if self.CLSW:args['class_weight']=self.class_weight_gsearch

        clf = GridSearchCV(svm.SVC(), args, scoring='roc_auc',cv=num_folds)
        clf.fit(self.xtrain,self.labels_trbi)
        clf.best_params_['probability']=True
        model=svm.SVC(**clf.best_params_)
        tt =time.time()-t
        if self.CLSW:results.extend([clf.best_params_["C"],clf.best_params_["gamma"],'','',clf.best_params_["class_weight"]])
        else:results.extend([clf.best_params_["C"],clf.best_params_["gamma"],'','',None])

    elif param == 'Optunity':
        def setSVM(C=None, G=None, W=None):
            if W:W={0:W}
            model = svm.SVC(kernel='rbf',C=10**C, gamma=10**G,class_weight=W,probability=True)
            return model

        @optunity.cross_validated(x=self.xtrain, y=self.ytrain, num_folds=num_folds, num_iter=num_iter)
        def svm_acc(x_train, y_train, x_test, y_test,**kargs):
            model=setSVM(**kargs)
            model.fit(x_train, y_train)
            # predict the test set
            predictions = model.decision_function(x_test)
            return optunity.metrics.roc_auc(y_test, predictions, positive=True)

        args = {'C':self.search['algorithm']['SVM']['kernel']['rbf']['C'],\
                'G':self.search['algorithm']['SVM']['kernel']['rbf']['gamma']}
        if self.CLSW:args['W']=self.search['algorithm']['SVM']['kernel']['rbf']['class_weight']

        optimal_pars, info, optimal_pars3 = optunity.maximize_structured(svm_acc,\
                                search_space=args,num_evals=num_evals)

        #optimal_pars, info, optimal_pars3 = optunity.maximize_structured(svm_acc,\
        #                        search_space=args,num_evals=num_evals,pmap=optunity.pmap)

        """
        #optimal_pars, info, optimal_pars3 = optunity.maximize(svm_acc,pmap=optunity.pmap,**args)
        #optimal_pars, info, optimal_pars3 = optunity.maximize(svm_acc, C=[-5, 2], gamma=[-5, 2],w=[1,10], pmap=optunity.pmap)
        """
        c,gm=10**optimal_pars["C"],10**optimal_pars["G"]
        clsw=optimal_pars["W"] if 'W' in optimal_pars else None
        model=svm.SVC(kernel='rbf',C=c,gamma=gm,probability=True)
        if not clsw==None:
            model=svm.SVC(kernel='rbf',C=c,gamma=gm,class_weight={0:clsw},probability=True)

        tt = info.stats["time"]
        results.extend([c,gm,'','',clsw])

        optima=max(info.call_log['values'])
        num_gen=optimal_pars3['num_generations']
        num_par=optimal_pars3['num_particles']
        Cs=info.call_log['args']['C']
        Gs=info.call_log['args']['G']
        Vs=info.call_log['values']

        # drawing how params evolved
        xys=[np.array([10**np.array(Cs),10**np.array(Gs)])]
        plotScatter(xys,'C','gamma',self.bname+'params',dir_o=self.dir_output_dgt,tnam=param)

        # drawing how optima evolved
        x=np.linspace(1,len(Vs),len(Vs))
        xys=[np.array([x,Vs])]
        plotScatter(xys,'Index','optimum',self.bname+'_optimum',dir_o=self.dir_output_dgt,tnam=param,cl=False)

        # export all results
        self.exportResults_csv(np.array([Cs,Gs,Vs]).transpose(),self.bname)
        #self.exportResults_pkl(info,self.bname+'svmrbf'+ ('_weight' if self.CLSW else ''))
        self.exportResults_pkl(info.call_log,self.bname+'svmrbf'+ ('_weight' if self.CLSW else ''))





    print '   %s: %0.2f sec'%(param,tt)
    #print '    C:%0.2f, g:%0.2f, w:%d'%(float(results[-3]),float(results[-2]),float(results[-1]))
    print '    C:%0.2f, g:%0.2f, w:%s'%(float(results[-5]),float(results[-4]),results[-1])
    results.extend(['','',num_folds,num_iter,num_gen,num_par,num_evals,optima,tt])
    return model,results
