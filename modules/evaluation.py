# encoding: utf-8
#-------------------------------------------------------------------------------
# Name:        evaluation
# Purpose:
#
# Author:      KSUZUKI
#
# Created:     14/03/2015
# Copyright:   (c) KSUZUKI 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import os,time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve,auc,confusion_matrix,accuracy_score,classification_report,roc_curve
from sklearn import cross_validation


# ROC curve drawing
def multipleROC(dictionary,odir,name):
    for data in dictionary.iterkeys():
        args=[[],[],[]]
        names=[]
        dict2 = dictionary[data]
        for param in dict2.iterkeys():
            names.append(param)
            for j in range(0,3):
                args[j].append(dict2[param][j])
        plotROC(args[0],args[1],args[2], os.path.join(odir,name+data+'_roc'),labels=names,multi=True)


def plotROC(fpr,tpr,roc_auc,name,multi=False,labels=''):
    if(len(plt.get_fignums())>0):plt.close()
    plt.figure()
    if multi:
        for i in range(0,len(fpr)):
            plt.plot(fpr[i], tpr[i], label='%s (area = %0.2f)' % (labels[i],roc_auc[i]))
    else:
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of %s'%name.split('\\')[-1])
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig((name+".png"), dpi=100)
    plt.close()

class Evaluation():

    def __init__(self,labels_true,ypredict,ypredict_score,dir_output,name,results,target_names):
        self.labels_true=labels_true
        self.ypredict=ypredict
        self.ypredict_score = ypredict_score
        self.dir_output= dir_output
        self.name = name
        self.results = results

    def evMain(self):
        self.ev_ROC()
        self.ev_precisionRecall()
        self.ev_accurracyRate()
        self.ev_confusionMatrix()
        #self.ev_report(target_names)

        return self.results,np.array(self.roc)


    def ev_ROC(self):
        #self.fpr, tpr, thresholds = roc_curve(self.labels_true, self.ypredict_score[:, 1], pos_label=1)
        self.fpr, tpr, thresholds = roc_curve(self.labels_true, self.ypredict_score[:, 0], pos_label=1)
        if np.isnan(self.fpr)[0] == False: # check if NaN
            roc_auc=auc(self.fpr,tpr)
            #plotROC(self.fpr,tpr,roc_auc,os.path.join(self.dir_output,self.name+'_roc'))
            self.roc = [self.fpr,tpr,roc_auc]
        else: roc_auc='NAN'
        ##########################################3
        self.results.append(str(roc_auc))

    # Precision-Recall　
    def ev_precisionRecall(self):
        #precision, recall, thresholds = precision_recall_curve(self.labels_true, self.ypredict_score[:, 1],pos_label=1)
        precision, recall, thresholds = precision_recall_curve(self.labels_true, self.ypredict_score[:, 0],pos_label=1)
        if np.isnan(self.fpr)[0] == False:
            area = auc(recall, precision)
            #pre_recall(labels_true, ypredict_score[:, 1],os.path.join(dir_output,name+'_pr'))
        else: area='NAN'
        self.results.append(str(area))

    # Accuracy rate
    def ev_accurracyRate(self):
        ac=accuracy_score(self.labels_true, self.ypredict)
        self.results.append(str(ac))

    # Confusion Matrix
    def ev_confusionMatrix(self):
        cm = confusion_matrix(self.labels_true, self.ypredict)
        for i in cm.flatten():
            self.results.append(str(i))
        print '       %d, %d'%(cm[0][0],cm[0][1])
        print '       %d, %d'%(cm[1][0],cm[1][1])

    #Classification Report--Precision、Recall, F-value
    def ev_report(target_names):
        cr = classification_report(self.labels_true, self.ypredict, target_names=[str(target_names[0]),str(target_names[1])])
        #cr = ((classification_report(labels_true, ypredict, target_names=[str(target_names[0]),str(target_names[1])])).replace('\n',',')).replace(' ',',').split(',')
        #for i in [39,45,51,57,73,79,85,91,102,108,114,120]:
        #    results=appendRList(results,cr[i])
        #print cr
        #results=appendRList(results,str(cr))

