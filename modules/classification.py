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

from modules.preposs import *
from modules.pca import *
from modules.ccode import *
from modules.evaluation import *
from modules.gabor import *
from modules.gabor_cv import *
from modules.gabor_opt1 import *


opts = ['Default','CV','Gsearch','Optunity']


class Classification():

    def main(self,dgts,featureExtr,tt):
        self.setParams()
        for i in range(dgts[0],dgts[1]):
            print "\n%s, digit %d "%(self.dataName,i)
            # file setting
            f = open(self.csvname, 'ab+')
            self.csvWriter = csv.writer(f)
            dir_output_dgt =os.path.join(self.dir_output,str(i))
            if not os.path.exists(dir_output_dgt):os.mkdir(dir_output_dgt)
            # initialization
            self.basicResults=[]
            self.bname = self.dataName+'_'+featureExtr+'_'+str(i)+'_'
            self.basicResults.extend([self.dataName,self.numEx,featureExtr,tt])
            # set Target digit
            self.setTarget(i)
            # Classification
            for param in opts:
                #model,results = self.buildSVM(param)
                self.buildModel()
                for data in ['test','train']:
                    results_=self.evaluateModel(model,copy.copy(results),param,data=data)
                    # export
                    results_.append(self.fparams)
                    self.csvWriter.writerow(results_)
            # drawing graph
            multipleROC(self.ROCs,dir_output_dgt,self.bname)
            del self.csvWriter
            f.close()

    def setParams(self):
        self.cvnum=10
        self.crange = np.logspace(0,2,num=15,base=10.0)
        self.grange = np.logspace(0,2,num=15,base=10.0)

        self.search = {'algorithm': {'k-nn': {'n_neighbors': [1, 5]},
                                'SVM': {'kernel': {'linear': {'C': [0, 2]},
                                                   'rbf': {'gamma': [0, 1], 'C': [0, 10]},
                                                   'poly': {'degree': [2, 5], 'C': [0, 50], 'coef0': [0, 1]}
                                                   }
                                        },
                                'naive-bayes': None,
                                'random-forest': {'n_estimators': [10, 30],
                                                  'max_features': [5, 20]}
                                }
                 }


    def setTarget(self,targetNumber):
        #name = self.bname.split('_')[0]+'_'+self.bname.split('_')[2]

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


    def buildSVM(self,param):
        print '\n   '+param
        tt,t=0,time.time()

        if np.min(self.cvnum<np.bincount(self.labels_vlbi))<self.cvnum:self.cvnum=np.min(np.bincount(self.labels_vlbi))
        self.ROCs = {}
        results = copy.copy(self.basicResults)
        results.append(param)

        if param == 'Default':
            model =svm.SVC(kernel='rbf',probability=True)
            results.extend([model.C,model.gamma,0.0])

        elif param=='CV':
            av=0
            for c in self.crange:
                for g in self.grange:
                    clf =svm.SVC(kernel='rbf',C=c,gamma=g,probability=True)
                    scores = cross_validation.cross_val_score(clf,self.xval,self.labels_vlbi,cv=self.cvnum, scoring='roc_auc')
                    if(av<np.mean(scores)):
                        av=np.mean(scores)
                        model=svm.SVC(kernel='rbf',C=c,gamma=g,probability=True)
                        tmp=[c,g]
            tt=time.time()-t
            tmp.append(tt)
            results.extend(tmp)

        elif param == 'Gsearch':
            args = {'kernel': ['rbf'], 'C':self.crange, 'gamma':self.grange}
            clf = GridSearchCV(svm.SVC(), args, scoring='roc_auc',cv=self.cvnum)
            clf.fit(self.xval,self.labels_vlbi)
            clf.best_params_['probability']=True
            model=svm.SVC(**clf.best_params_)
            tt =time.time()-t
            results.extend([clf.best_params_["C"],clf.best_params_["gamma"],tt])

        elif param == 'Optunity':
            # set score function: twice iterated 10-fold cross-validated accuracy
            @optunity.cross_validated(x=self.xtrain, y=self.ytrain, num_folds=10, num_iter=2)
            def svm_acc(x_train, y_train, x_test, y_test, C, gamma):
                model = svm.SVC(kernel='rbf',C=C, gamma=gamma,probability=True).fit(x_train, y_train)
                y_pred = model.predict(x_test)
                return optunity.metrics.accuracy(y_test, y_pred)

            args = {'num_evals':100, 'C':[min(self.crange),max(self.crange)], 'gamma':[min(self.grange),max(self.grange)]}
            optimal_pars, optimal_pars2, _ = optunity.maximize(svm_acc, **args)
            optimal_pars['probability']=True
            optimal_pars['kernel']='rbf'

            model =svm.SVC(**optimal_pars)
            tt = optimal_pars2.stats["time"]
            results.extend([optimal_pars["C"],optimal_pars["gamma"],tt])
        print '   %s: %0.2f sec'%(param,tt)
        print '    C:%0.2f, g:%0.2f'%(float(results[-3]),float(results[-2]))
        return model,results

    def buildModel(self):
        @optunity.cross_validated(x=self.xtrain, y=self.ytrain, num_folds=10, num_iter=2)
        def performance(x_train, y_train, x_test, y_test,
                        algorithm, n_neighbors=None, n_estimators=None, max_features=None,
                        kernel=None, C=None, gamma=None, degree=None, coef0=None):
            # fit the model
            if algorithm == 'k-nn':
                model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
                model.fit(x_train, y_train)
            elif algorithm == 'SVM':
                model = train_svm(x_train, y_train, kernel, C, gamma, degree, coef0)
            elif algorithm == 'naive-bayes':
                model = GaussianNB()
                model.fit(x_train, y_train)
            elif algorithm == 'random-forest':
                model = RandomForestClassifier(n_estimators=int(n_estimators),
                                               max_features=int(max_features))
                model.fit(x_train, y_train)
            else:
                raise ArgumentError('Unknown algorithm: %s' % algorithm)

            # predict the test set
            if algorithm == 'SVM':
                predictions = model.decision_function(x_test)
            else:
                predictions = model.predict_proba(x_test)[:, 1]

            return optunity.metrics.roc_auc(y_test, predictions, positive=True)


        optimal_configuration, info, _ = optunity.maximize_structured(performance,\
                                search_space=self.search,num_evals=300)
        #print(optimal_configuration)
        #print(info.optimum)
        #{'kernel': 'poly', 'C': 38.5498046875, 'algorithm': 'SVM', 'degree': 3.88525390625, 'n_neighbors': None, 'n_estimators': None, 'max_features': None, 'coef0': 0.71826171875, 'gamma': None}
        #0.979302949566
        #Finally, lets make the results a little bit more readable. All dictionary items in optimal_configuration with value None can be removed.

        solution = dict([(k, v) for k, v in optimal_configuration.items() if v is not None])
        print('Solution\n========')
        print("\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items())))


    def evaluateModel(self,model,results,param,data='test'):
        print '     '+data
        name=self.bname+data+'_'+param
        results.insert(14,data)
        if data == 'test':
            xpredict,labels_true=self.xtest,self.labels_tebi
        elif data=='train':
            xpredict,labels_true=self.xtrain,self.labels_trbi

        # fit to the test data
        optimal_model = model.fit(self.xtrain, self.ytrain)
        ypredict = optimal_model.predict(xpredict)
        ypredict_score = optimal_model.decision_function(xpredict)

        # Evaluate model
        e=Evaluation(labels_true,ypredict,ypredict_score,self.dir_output,name,results,self.target_names)
        results,roc=e.evMain()
        if not self.ROCs.has_key(data): self.ROCs[data]={param:roc}
        else:self.ROCs[data].update({param:roc})

        return results



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




