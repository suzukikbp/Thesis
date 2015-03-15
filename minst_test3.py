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




import os,cv2,time
import numpy as np
import optunity,optunity.metrics
import pylab

from modules.preposs import *
from modules.pca import *
from modules.cvfunc import *
from modules.evaluation import *



dataSets = ['mnist_basic','mnist_background_images','mnist_background_random','mnist_rotation','mnist_rotation_back_image','mnist_rotation_new']
featureExtrs = ['PCA','Bluring']

def main(targetNumber):

    #################################
    # set Parameters
    pixel = 28 # pixel size (only for a square)
    dir_input = r"C:\Users\KSUZUKI\Dropbox\02_Research\03_MNIST\icml2007\data"
    dir_output = r"C:\Users\KSUZUKI\Dropbox\02_Research\03_MNIST\graphs"
    dataSet =dataSets[0]
    featureExtr = featureExtrs[1]
    smallTest = 0 # 0: full dataset, 1: small fraction of dataset



    #################################
    # 1. downloade data
    #obtain(dir_input,dataSet)

    #################################
    # 2. make Varidation data
    #makeVarid(dir_input,dataSet)

    #################################
    # 3-1. set data in memory
    datas = load(dir_input,True,dataSet,pixel)

    # training data
    images_tr = datas["train"][0].mem_data[0]
    labels_tr = datas["train"][0].mem_data[1]
    # test data
    images_te = datas["test"][0].mem_data[0]
    labels_te = datas["test"][0].mem_data[1]

    # 3-2. extract the specific number based on "target_names"
    target_names =([targetNumber],np.delete(np.linspace(0,9,10,np.int16),targetNumber))
    if(smallTest == 1):target_names =([targetNumber],[8])
    drawimg(images_tr,labels_tr,dir_output,(dataSet+str(target_names[0])+"__test"),pixel)
    images_tr,labels_tr,labels_trbi= selectNum(images_tr,labels_tr,target_names)
    images_te,labels_te,labels_tebi= selectNum(images_te,labels_te,target_names)
    print "#trainImage : %d (#target : %d, #rest : %d )"%(len(labels_tr),len(labels_trbi[labels_trbi==0]),len(labels_trbi[labels_trbi==1]))
    print "#testImage : %d"%len(labels_te)


    #drawimg(images_tr,labels_tr,dir_output,dataSet,pixel)
    drawimg(images_tr,labels_trbi,dir_output,(dataSet+str(target_names[0])),pixel)

    #################################
    # 4-1. extract characteristics-- PCA
    if(featureExtr=='PCA'):
        xdat,xtest,score = pca(images_tr,images_te,target_names,pixel)
        plotpca(target_names,xdat,labels_tr,dataSet,dir_output)


    # 4-2. extract characteristics-- contour
    elif(featureExtr=='Bluring'):
        tt = time.time()
        print "  Contour/Blur"

        # training data
        # conversion into binary image
        _,thresh_tr = makeBinary(images_tr.copy())
        drawimg(thresh_tr.copy(),labels_tr.copy(),dir_output,(dataSet+"_binary"),pixel)
        # directional decomposition of image into directional sub-images
        xdat,contours = makeContor(thresh_tr,pixel)
        # convolution and sampling the directional sub-images

        # test data
        ##_,thresh_te = makeBinary(images_te.copy()[0:25000])
        _,thresh_te = makeBinary(images_te.copy())
        xtest,_= makeContor(thresh_te,pixel)
        # drawing
        drawimgContour(images_tr,labels_tr,contours,dir_output,(dataSet+'_bluring'),pixel)
        print "  Contour/Blur: %0.2f sec" %((time.time()-tt))


    #################################
    # 5. score function: twice iterated 10-fold cross-validated accuracy
    tt = time.time()
    print "  Tuning"
    ydat =labels_trbi

    @optunity.cross_validated(x=xdat, y=ydat, num_folds=10, num_iter=2)
    def svm_acc(x_train, y_train, x_test, y_test, C, gamma):
        model = svm.SVC(kernel='rbf',C=C, gamma=gamma,probability=True).fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return optunity.metrics.accuracy(y_test, y_pred)


    #################################
    # 6. perform tuning
    #args = {'num_evals':100, 'C':[0, 10], 'gamma':[0, 1],'pmap':'optunity.pmap'}
    args = {'num_evals':100, 'C':[0, 10], 'gamma':[0, 1]}
    optimal_pars, optimal_pars2, _ = optunity.maximize(svm_acc, **args)
    #optimal_pars, optimal_pars2, _ = optunity.maximize(svm_acc, num_evals=100, C=[0, 10], gamma=[0, 1],pmap=optunity.pmap)
    optimal_pars['probability']=True
    optimal_pars['kernel']='rbf'
    print'  Optimal params C: %f, gamma: %f'%(optimal_pars["C"],optimal_pars["gamma"])
    print "  Tuning: %0.2f sec" % optimal_pars2.stats["time"]


    #################################
    # 7 train model with tuned/default hyperparameters
    for i in range(0,4):

        predictImages = images_te
        predictData = xtest
        model =svm.SVC(**optimal_pars)
        kind = dataSet+"_predict_test_optunity"
        labels =labels_tebi

        if(i == 1):
            kind = dataSet+"_predict_train_optunity"
            predictData =xdat
            predictImages = images_tr
            labels =labels_trbi
        elif(i == 2):
            kind = dataSet+"_predict_test_default"
            model =svm.SVC(kernel='rbf',probability=True)
        elif(i == 3):
            kind = dataSet+"_predict_train_default"
            model =svm.SVC(kernel='rbf',probability=True)
            predictData =xdat
            predictImages = images_tr
            labels =labels_trbi

        print kind
        optimal_model = model.fit(xdat, ydat)
        # fit to the test data
        predict = optimal_model.predict(predictData)
        predict_prob = optimal_model.predict_proba(predictData)
        drawimg(predictImages,predict,dir_output,kind,pixel)


        #################################
        # 8. evaluation--
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        print confusion_matrix(labels, predict)

        # Accuracy rate
        from sklearn.metrics import accuracy_score
        print  "  Accuracy rate: %0.2f" % accuracy_score(labels, predict)

        #Classification Report--Precision、Recall, F-value
        from sklearn.metrics import classification_report
        print(classification_report(labels, predict, target_names=[str(target_names[0]),str(target_names[1])]))

        # Compute Precision-Recall and plot curve
        pre_recall(labels, predict_prob[:, 1],kind)

        # ROC curve
        roc(labels, predict_prob[:, 1],kind)



    """
    # 7-1. train model on the full training set with tuned hyperparameters
    model_optunity =svm.SVC(**optimal_pars)
    optimal_model_optunity = model_optunity.fit(xdat, ydat)
    predict_optunity = optimal_model_optunity.predict(xtest)
    predict_optunity_prob = optimal_model_optunity.predict_proba(xtest)
    # drawing
    drawimg(images_te,predict_optunity,dir_output,(dataSet+"_predict_optunity"),pixel)


    # 7-2. train model on the full training set with default hyperparameters
    #model_default =svm.SVC(kernel='rbf',probability=True,C=3.0, gamma=1.0)
    model_default =svm.SVC(kernel='rbf',probability=True)
    optimal_model_default = model_default.fit(xdat, ydat)
    predict_default = optimal_model_default.predict(xtest)
    predict_default_prob = optimal_model_default.predict_proba(xtest)
    # drawing
    drawimg(images_te,predict_default,dir_output,(dataSet+"_predict_default"),pixel)


    # 8. evaluation--
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    print confusion_matrix(labels_tebi, predict_optunity)
    print confusion_matrix(labels_tebi, predict_default)

    # Accuracy rate
    from sklearn.metrics import accuracy_score
    print  "  Accuracy rate: %0.2f" % accuracy_score(labels_tebi, predict_optunity)
    print  "  Accuracy rate: %0.2f" % accuracy_score(labels_tebi, predict_default)

    #Classification Report--Precision、Recall, F-value
    from sklearn.metrics import classification_report
    print(classification_report(labels_tebi, predict_optunity, target_names=[str(target_names[0]),str(target_names[1])]))
    print(classification_report(labels_tebi, predict_default, target_names=[str(target_names[0]),str(target_names[1])]))

    # Compute Precision-Recall and plot curve
    pre_recall(labels_tebi, predict_optunity_prob[:, 1],'optunity')
    pre_recall(labels_tebi, predict_default_prob[:, 1],'default')

    # ROC curve
    roc(labels_tebi, predict_optunity_prob[:, 1],'optunity')
    roc(labels_tebi, predict_default_prob[:, 1],'default')
    """


    print "\n\nfinish"



if __name__ == '__main__':
    for i in range(0,9):
        print "Start : %d "%i
        main(i)

