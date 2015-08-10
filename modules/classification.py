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
# k nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
# support vector machine classifier
from sklearn.svm import SVC
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Random Forest
from sklearn.ensemble import RandomForestClassifier

from modules.preposs import *
from modules.pca import *
from modules.ccode import *
from modules.evaluation import *
from modules.gabor import *
from modules.gabor_cv import *
from modules.gabor_opt1 import *




class Classification():
    def main(self,dgts,featureExtr,tt,mcl):
        self.mutipleClassifiers=mcl
        opts = ['Default','CV','Gsearch','Optunity']
        if self.mutipleClassifiers:opts=['Optunity_mcl']
        self.setParams()
        for i in range(dgts[0],dgts[1]):
            print "\n%s, digit %d "%(self.dataName,i)
            # file setting
            self.ROCs = {}
            f = open(self.csvname, 'ab+')
            self.csvWriter = csv.writer(f)
            dir_output_dgt =os.path.join(self.dir_output,str(i))
            if not os.path.exists(dir_output_dgt):os.mkdir(dir_output_dgt)
            # initialization
            self.basicResults=[]
            if self.mutipleClassifiers:
                self.bname=self.dataName+'_'+featureExtr+'_'+str(i)+'_mcl'+'_'
            else:
                self.bname=self.dataName+'_'+featureExtr+'_'+str(i)+'_'

            self.basicResults.extend([self.dataName,self.numEx,featureExtr,tt])
            # set Target digit
            self.setTarget(i)
            # Classification
            for param in opts:
                if self.mutipleClassifiers:
                    model,results = self.buildModel(copy.copy(self.basicResults))
                else:
                    model,results = self.buildSVM(param,copy.copy(self.basicResults))
                for data in ['test','train']:
                    results_,_=self.evaluateModel(model,copy.copy(results),param,data=data)
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
        self.numFeature= self.xtrain.shape[1]

        self.search = {'algorithm': {'k-nn': {'n_neighbors': [1, 5]},
                                'SVM': {'kernel': {'linear': {'C': [0, 2]},
                                                   'rbf': {'gamma': [0, 1], 'C': [0, 10]},
                                                   'poly': {'degree': [2, 5], 'C': [0, 50], 'coef0': [0, 1]}
                                                   }
                                        },
                                'naive-bayes': None,
                                'random-forest': {'n_estimators': [10, 30],'max_features':[2,self.numFeature]}
                                }
                 }


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


    def buildSVM(self,param,results):
        print '\n   '+param
        tt,t=0,time.time()

        if np.min(self.cvnum<np.bincount(self.labels_vlbi))<self.cvnum:self.cvnum=np.min(np.bincount(self.labels_vlbi))
        results.extend(['SVM_rbf',param])

        if param == 'Default':
            model =svm.SVC(kernel='rbf',probability=True)
            results.extend([model.C,model.gamma])

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
            results.extend(tmp)

        elif param == 'Gsearch':
            args = {'kernel': ['rbf'], 'C':self.crange, 'gamma':self.grange}
            clf = GridSearchCV(svm.SVC(), args, scoring='roc_auc',cv=self.cvnum)
            clf.fit(self.xval,self.labels_vlbi)
            clf.best_params_['probability']=True
            model=svm.SVC(**clf.best_params_)
            tt =time.time()-t
            results.extend([clf.best_params_["C"],clf.best_params_["gamma"]])

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
            results.extend([optimal_pars["C"],optimal_pars["gamma"]])

        print '   %s: %0.2f sec'%(param,tt)
        print '    C:%0.2f, g:%0.2f'%(float(results[-2]),float(results[-1]))
        results.extend(['','','','','',-1,tt])
        return model,results


    def setModel(self,algorithm=None, n_neighbors=None, n_estimators=None, max_features=None,
                        kernel=None, C=None, gamma=None, degree=None, coef0=None):
        params=[algorithm+'_'+str(kernel),'Optunity_mcl',C,gamma,degree,coef0,n_neighbors,n_estimators,max_features]
        if algorithm == 'k-nn':
            model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
        elif algorithm == 'SVM':
            model = self.train_svm(kernel, C, gamma,degree,coef0)
        elif algorithm == 'naive-bayes':
            model = GaussianNB()
        elif algorithm == 'random-forest':
            model = RandomForestClassifier(n_estimators=int(n_estimators),max_features=int(max_features))
        else:
            raise ArgumentError('Unknown algorithm: %s' % algorithm)
            return None
        return model,params

    """A generic SVM training function, with arguments based on the chosen kernel."""
    def train_svm(self,kernel, C, gamma, degree, coef0):
        if kernel == 'linear':
            model = SVC(kernel=kernel, C=C)
        elif kernel == 'poly':
            model = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)
        elif kernel == 'rbf':
            model = SVC(kernel=kernel, C=C, gamma=gamma)
        else:
            raise ArgumentError("Unknown kernel function: %s" % kernel)
        return model


    def buildModel(self,results):
        #self.ROCs = {}
        tt=time.time()
        @optunity.cross_validated(x=self.xtrain, y=self.ytrain, num_folds=2, num_iter=2)
        def performance(x_train, y_train, x_test, y_test,**kargs):
            model,_=self.setModel(**kargs)
            model.fit(x_train, y_train)
            # predict the test set
            if kargs['algorithm'] == 'SVM':
                predictions = model.decision_function(x_test)
            else:
                predictions = model.predict_proba(x_test)[:, 1]

            return optunity.metrics.roc_auc(y_test, predictions, positive=True)


        optimal_configuration, info, _ = optunity.maximize_structured(performance,\
                                search_space=self.search,num_evals=50)

        solution = dict([(k, v) for k, v in optimal_configuration.items() if v is not None])
        print("\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items())))

        self.restructInfo(info,results)
        model,params=self.setModel(**solution)
        results.extend(params)
        #print time.time()-tt
        results.extend([max(info.call_log['values']),time.time()-tt])
        return model,results


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
        if 'SVM' in results[10]:
            ypredict_score = model.decision_function(xpredict)
        else:
            ypredict_score = model.predict_proba(xpredict)

        # Evaluate model
        e=Evaluation(labels_true,ypredict,ypredict_score,self.dir_output,name,results,self.target_names)
        results,roc=e.evMain()
        if self.mutipleClassifiers==False:
            if not self.ROCs.has_key(data): self.ROCs[data]={param:roc}
            else:self.ROCs[data].update({param:roc})

        return results,roc

    def restructInfo(self,info,results):

        algos=list(set(info.call_log['args']['algorithm']))
        for alg in algos:
            result =copy.copy(results)
            alg_idx = np.where(np.array(info.call_log['args']['algorithm'])==alg)[0]
            values=np.array([info.call_log['values'][i] for i in alg_idx])
            idx_max=np.where(values==max(values))[0][0]

            ker=str(np.array([info.call_log['args']['kernel'][i] for i in alg_idx])[idx_max])
            dics={'algorithm':alg,'kernel':ker}
            if ker==None: algoname=alg
            else:algoname=alg+'_'+ker
            result.extend([algoname,'Optunity_msl'])
            for key in['C','gamma','degree','coef0','n_neighbors','n_estimators','max_features']:
                pars= np.array([info.call_log['args'][key][i] for i in alg_idx])
                result.append(pars[idx_max])
                dics[key]=pars[idx_max]
            result.extend([max(values),'time'])
            model,_=self.setModel(**dics)
            for data in ['test','train']:
                results_,roc=self.evaluateModel(model,copy.copy(result),'Optunity_msl',data=data)
                results_.append(self.fparams)
                self.csvWriter.writerow(results_)

                if self.mutipleClassifiers:
                    if not self.ROCs.has_key(data): self.ROCs[data]={algoname:roc}
                    else:self.ROCs[data].update({algoname:roc})


        return

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




