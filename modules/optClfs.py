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

"""A generic SVM training function, with arguments based on the chosen kernel."""
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

def optClf(self,results,num_folds=2, num_iter=10,num_evals=100):
    tt=time.time()
    @optunity.cross_validated(x=self.xtrain, y=self.ytrain, num_folds=num_folds, num_iter=num_iter)
    def performance(x_train, y_train, x_test, y_test,**kargs):
        model,_=setModel(**kargs)
        model.fit(x_train, y_train)
        # predict the test set
        if kargs['algorithm'] == 'SVM':
            predictions = model.decision_function(x_test)
        else:
            predictions = model.predict_proba(x_test)[:, 1]
        #return optunity.metrics.roc_auc(y_test, predictions, positive=True)
        return optunity.metrics.roc_auc(y_test, predictions)

    optimal_configuration, info, optimal_pars3 = optunity.maximize_structured(performance,\
                            search_space=self.search,num_evals=num_evals)

    solution = dict([(k, v) for k, v in optimal_configuration.items() if v is not None])
    print("\n".join(map(lambda x: "  %s: %s" % (x[0], str(10**x[1]) if x[0]in['C','gamma'] else str(x[1]) ), solution.items())))
    num_gen=optimal_pars3['num_generations']
    num_par=optimal_pars3['num_particles']

    Vs=info.call_log['values']
    x=np.linspace(1,len(Vs),len(Vs))
    xys=[np.array([x,Vs])]
    plotScatter(xys,'Index','optimum',self.bname+'optimum',dir_o=self.dir_output_dgt,cl=False)

    As=np.array(info.call_log['args']['algorithm'],dtype='object')
    Ks=np.array(info.call_log['args']['kernel'],dtype='object')
    Ks[np.where(Ks.astype(str)=='None')]=''
    #algos=np.add(As,Ks).astype(str)
    algos=np.add(As,'' if Ks is None else Ks).astype(str)

    #Gs=sorted(np.unique(algos))
    #for i,v in enumerate(Gs):
    #    self.Algos[np.where(Algos==v)]=i

    for i,v in enumerate(self.clfs):
        algos[np.where(algos==v)]=i

    x=np.linspace(1,len(Ks),len(Ks))
    xys=[np.array([x,algos.astype(np.int64)])]
    plotScatter(xys,'Index','optimum',self.bname+'algos',dir_o=self.dir_output_dgt,cl=False)

    # export results
    #self.exportResults_csv(np.array([Cs,Gs,Vs]).transpose(),self.bname)
    self.exportResults_pkl(info.call_log,self.bname+'svmrbf'+ ('_weight' if self.CLSW else ''))


    pso_params=[num_folds,num_iter,num_gen,num_par,num_evals]
    restructInfo(self,info,results,pso_params)
    model,params=setModel(**solution)
    results.extend(params)
    results.extend(pso_params)
    results.extend([max(info.call_log['values']),time.time()-tt])

    return model,results


# function to see each
def restructInfo(self,info,results,pso_params):

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
        print "  %s"%algoname
        result.extend([algoname,'Optunity_msl'])
        searchspace=['C','gamma','degree','coef0','class_weight','n_estimators','max_features']
        for key in searchspace:
            if key in info.call_log['args'].keys():
                pars= np.array([info.call_log['args'][key][i] for i in alg_idx])
                #pars= np.array([[info.call_log['args'][key][i] for i in alg_idx] if info.call_log['args'][key][i] in alg_idx else ''])
                result.append(pars[idx_max])
                dics[key]=pars[idx_max]
            else:
                result.append('')
        result.extend(pso_params)
        result.extend([max(values),'time'])

        model,_=setModel(**dics)
        for data in ['test','train']:
            results_,roc=self.evaluateModel(model,copy.copy(result),'Optunity_mcl_info',data=data)
            results_.append(self.fparams)
            self.csvWriter.writerow(results_)

            if not self.ROCs.has_key(data): self.ROCs[data]={algoname:roc}
            else:self.ROCs[data].update({algoname:roc})
