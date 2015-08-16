# encoding: utf-8

#-------------------------------------------------------------------------------
# Name:        mnist
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from modules.preposs import *
from modules.pca import *
from modules.ccode import *
from modules.evaluation import *
from modules.gabor import *
from modules.gabor_cv import *
from modules.gabor_opt1 import *
from modules.optClfs import *
from modules.optSVM import *


class Classification():
    def main(self,dgts,featureExtr,tt,modelchoice='optSVM',classweight=True,DEBUG=0):
        self.DEBUG=DEBUG
        self.CLSW=classweight
        opts = ['Default','Gsearch','Optunity']

        for i in range(dgts[0],dgts[1]):
            print "\n%s, digit %d "%(self.dataName,i)
            # file setting
            f = open(self.csvname, 'ab+')
            self.csvWriter = csv.writer(f)
            self.dir_output_dgt =os.path.join(self.dir_output,str(i))
            if not os.path.exists(self.dir_output_dgt):os.mkdir(self.dir_output_dgt)
            # initialization
            self.ROCs = {}
            self.basicResults=[self.dataName,self.numEx,featureExtr,tt,self.CLSW]
            # set Target digit
            self.setTarget(i)
            self.setParams()

            # Classification
            if modelchoice=='optClfs':
                self.bname=self.dataName+'_'+featureExtr+'_'+str(i)+'_clfs'+'_'
                model,results = optClf(self,copy.copy(self.basicResults))
                for data in ['test','train']:
                    results_,_=self.evaluateModel(model,copy.copy(results),modelchoice,data=data)
                    # export
                    results_.append(self.fparams)
                    self.csvWriter.writerow(results_)

            elif modelchoice=='optSVM':
                self.bname=self.dataName+'_'+featureExtr+'_'+str(i)+'_'
                for param in opts:
                    model,results = optSVM(self,param,copy.copy(self.basicResults))
                    for data in ['test','train']:
                        results_,_=self.evaluateModel(model,copy.copy(results),param,data=data)
                        # export
                        results_.append(self.fparams)
                        self.csvWriter.writerow(results_)

            # drawing graph
            multipleROC(self.ROCs,self.dir_output_dgt,self.bname)
            del self.csvWriter
            f.close()

    def setParams(self):
        self.crange = np.logspace(-5,2,num=15,base=10.0)
        self.grange = np.logspace(-5,2,num=15,base=10.0)
        self.wrange = np.linspace(1,10,10)
        self.num_folds=3
        self.num_iter=10
        self.num_evals=100

        if self.DEBUG:
            self.num_folds=2
            self.num_iter=5

        self.numFeature= self.xtrain.shape[1]
        if np.min(np.bincount(self.labels_trbi))<self.num_folds:
            self.num_folds=np.min(np.bincount(self.labels_trbi))

        if self.CLSW:
            self.class_weight_gsearch=[{0: w} for w in range(1,11)]
            self.search = {'algorithm': {'SVM': {'kernel': {'linear': {'C': [-5, 2],'class_weight':[1,10]},
                                               'rbf': {'gamma': [-5, 2], 'C': [-5, 2],'class_weight':[1,10]},
                                               'poly': {'degree': [2, 5], 'C': [-5,2], 'coef0': [0, 1],'class_weight':[1,10]}
                                               }
                                    },
                            #'k-nn': {'n_neighbors': [1, 5]},
                            'naive-bayes': None,
                            'random-forest': {'n_estimators': [10, 30],'max_features':[2,self.numFeature]}
                            }
             }

        else:
            self.search = {'algorithm': {'SVM': {'kernel': {'linear': {'C': [-5, 2]},
                                                       'rbf': {'gamma': [-5, 2], 'C': [-5, 2]},
                                                       'poly': {'degree': [2, 5], 'C': [-5,2], 'coef0': [0, 1]}
                                                       }
                                            },
                                    #'k-nn': {'n_neighbors': [1, 5]},
                                    'naive-bayes': None,
                                    'random-forest': {'n_estimators': [10, 30],'max_features':[2,self.numFeature]}
                                    }
                     }


        """
        if self.CLSW:class_weight=[1,10]
        else:class_weight=None
        self.search = {'algorithm': {'SVM': {'kernel': {'linear': {'C': [-5, 2],'class_weight':class_weight},
                                                   'rbf': {'gamma': [-5, 2], 'C': [-5, 2],'class_weight':class_weight},
                                                   'poly': {'degree': [2, 5], 'C': [-5,2], 'coef0': [0, 1],'class_weight':class_weight}
                                                   }
                                        },
                                #'k-nn': {'n_neighbors': [1, 5]},
                                'naive-bayes': None,
                                'random-forest': {'n_estimators': [10, 30],'max_features':[2,self.numFeature]}
                                }
                 }
        """

        #1:'random-forest'
        #2:'naive-bayes'
        #3:'SVMrbf'
        #4:'SVMpoly'
        #5:'SVMlinear'
        algos=map(lambda x: map(lambda y: x[0]+y[0],x[1]['kernel'].items()) if x[0] == 'SVM' else [x[0]], self.search['algorithm'].items())
        self.clfs=sorted([item for sublist in algos for item in sublist],reverse=True)


    def setTarget(self,targetNumber):
        # Extract the specific number based on "self.target_names"
        self.target_names =([targetNumber],np.delete(np.linspace(0,9,10,np.int16),targetNumber))
        # class(0:target, 1:rest of all)
        self.labels_trbi,self.labels_vlbi,self.labels_tebi= [selectNum(dat,self.target_names)for dat in [self.lab_tr_all,self.lab_vl_all,self.lab_te_all]]
        # set binary data to Y
        self.ytrain =self.labels_trbi

        self.basicResults.extend([self.target_names[0][0],-self.target_names[0][0],len(self.labels_trbi[self.labels_trbi==0]),len(self.labels_trbi[self.labels_trbi==1]),len(self.labels_tebi[self.labels_tebi==0]),len(self.labels_tebi[self.labels_tebi==1])])
        print "  #train:%d (%d: %d, %d: %d)"%(len(self.labels_trbi),self.target_names[0][0],len(self.labels_trbi[self.labels_trbi==0]),-self.target_names[0][0],len(self.labels_trbi[self.labels_trbi==1]))
        print "  #test: %d (%d: %d, %d: %d)"%(len(self.labels_tebi),self.target_names[0][0],len(self.labels_tebi[self.labels_tebi==0]),-self.target_names[0][0],len(self.labels_tebi[self.labels_tebi==1]))
        print "  #val: %d (%d: %d, %d: %d)"%(len(self.labels_vlbi),self.target_names[0][0],len(self.labels_vlbi[self.labels_vlbi==0]),-self.target_names[0][0],len(self.labels_vlbi[self.labels_vlbi==1]))


    def evaluateModel(self,model,results,param,data='test'):
        print '     '+data
        name=self.bname+data+'_'+param
        #results.insert(19,data)
        results.append(data)

        if data == 'test':
            xpredict,labels_true=self.xtest,self.labels_tebi
        elif data=='train':
            xpredict,labels_true=self.xtrain,self.labels_trbi

        model = model.fit(self.xtrain, self.ytrain)
        ypredict = model.predict(xpredict)
        if 'SVM' in results[11]:
            ypredict_score = model.decision_function(xpredict)
        else:
            ypredict_score = model.predict_proba(xpredict)[:, 1]

        # Evaluate model
        e=Evaluation(labels_true,ypredict,ypredict_score,self.dir_output,name,results,self.target_names)
        results,roc=e.evMain()
        if not self.ROCs.has_key(data): self.ROCs[data]={param:roc}
        else:self.ROCs[data].update({param:roc})

        return results,roc


    # normalization
    # note that negative value is possible
    def norm(self,imgs):
        imglists=[]
        for img in imgs:
            imglists.append(np.array(img.max()*((img - img.mean()) / img.std()),dtype=img.dtype))
        return np.array(imglists)

    def writeout(self,datas,dir_out):
        labels=["img_tr_all","lab_tr_all","img_val_all","lab_val_all","img_te_all","lab_te_all"]
        for i in range(0,len(datas)):
            data = datas[i]
            f = open(os.path.join(dir_output,labels[i]+'.csv'), 'wb')
            csvw = csv.writer(f)
            if i%2 == 0:
                for j in range(0,(len(data)-1)):
                    csvw.writerow(data[j])
            else:
                for j in range(0,(len(data)-1)):
                    csvw.writerow([data[j]])
            del csvw
            f.close()

    def createCombs(self,numgs,numps,nfols,nitrs):
        combs=[]
        for numg in numgs:
            for nump in numps:
                for nfol in nfols:
                    for nitr in nitrs:
                        combs.append([numg,nump,nfol,nitr])
        return combs




