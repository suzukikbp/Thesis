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


"""
Loading
  Loading : 0.9 sec
  Contour/Blur
  Contour/Blur: 858.54 sec
  Tuning
"""



import os,cv2,time
import numpy as np
import optunity,optunity.metrics
import pylab

from modules.preposs import *
from modules.pca import *
from modules.cvfunc import *



def main():

    arLen = 1000
    # pixel size(only for a square)
    pixel = 28
    #target_names =np.array([0,1,2,3,4,5,6,7,8,9])
    target_names =np.array([8,9])
    dir_dat = r"C:\Users\KSUZUKI\Dropbox\02_Research\03_MNIST\icml2007\data"
    dir_fig = r"C:\Users\KSUZUKI\Dropbox\02_Research\03_MNIST\graphs"

    kind ='mnist_basic'
    #kind ='mnist_background_images'
    #kind ='mnist_background_random'
    #kind ='mnist_rotation'
    #kind ='mnist_rotation_back_image'
    #kind ='mnist_rotation_back_image_new'
    #kind ='mnist_rotation_new.zip



    # 1. downloade data
    #obtain(dir_dat,kind)

    # 2. make Varidation data
    #makeVarid(dir_dat,kind)

    # 3-1. set data in memory
    datas = load(dir_dat,True,kind,pixel)
    #datas["train"][0].mem_data[0][9999][739]

    # training data
    images_tr = datas["train"][0].mem_data[0]
    labels_tr = datas["train"][0].mem_data[1]
    # test data
    images_te = datas["test"][0].mem_data[0]
    labels_te = datas["test"][0].mem_data[1]

    # 3-2. extract the specific number based on "target_names"
    images_tr,labels_tr= selectNum(images_tr,labels_tr,target_names)
    images_te,labels_te= selectNum(images_te,labels_te,target_names)

    # 3-3. make binary labels
    labels_trbi = labels_tr.copy()
    labels_trbi[np.where(labels_trbi==target_names[0])]=10
    labels_trbi[np.where(labels_trbi==target_names[1])]=11
    labels_trbi = labels_trbi-10
    labels_tebi = labels_te.copy()
    labels_tebi[np.where(labels_tebi==target_names[0])]=10
    labels_tebi[np.where(labels_tebi==target_names[1])]=11
    labels_tebi = labels_tebi-10

    drawimg(images_tr,labels_tr,dir_fig,kind,pixel)

    """
    # 4-1. extract characteristics-- PCA
    images_pca_tr,score = pca(images_tr,target_names,pixel)
    images_pca_te,score = pca(images_te,target_names,pixel)
    plotpca(target_names,images_pca_tr,labels_tr,kind,dir_fig)
    """


    # 4-2. extract characteristics-- sobel
    #order of the derivative of x and y is 1
    tt = time.time()
    print "  Sobel"
    images_slb_tr = makeSober(images_tr,pixel)
    images_slb_te = makeSober(images_te,pixel)
    # drawing
    drawimg(images_slb_tr,labels_tr,dir_fig,(kind+"_sobel"),pixel)
    gaussians_tr = makeContor(images_slb_tr,pixel)
    gaussians_te = makeContor(images_slb_te,pixel)


    print "  Sobel: %0.2f sec" %((time.time()-tt))

    """
    # 4-3. extract characteristics-- contour
    tt = time.time()
    print "  Contour/Blur"

    # training data
    # conversion into binary image
    _,thresh_tr = makeBinary(images_tr.copy())
    drawimg(thresh_tr.copy(),labels_tr.copy(),dir_fig,"binary_"+kind,pixel)
    # directional decomposition of image into directional sub-images
    #thresh_tr, conts_tr, hierarchys_tr, directs_tr = makeContor(thresh_tr,pixel)
    #directs_tr = makeContor(thresh_tr,pixel)
    gaussians_tr = makeContor(thresh_tr,pixel)
    # convolution and sampling the directional sub-images
    #gaussians_tr = direct2gaussian(directs_tr,pixel,5)

    # test data
    #_,thresh_te = makeBinary(images_te.copy())
    _,thresh_te = makeBinary(images_te.copy()[0:25000])
    #thresh_te, conts_te, hierarchys_te, directs_te = makeContor(thresh_te,pixel)
    gaussians_te = makeContor(thresh_te,pixel)
    #gaussians_te = direct2gaussian(directs_te,pixel,5)
    print "  Contour/Blur: %0.2f sec" %((time.time()-tt))
    """



    # 5. score function: twice iterated 10-fold cross-validated accuracy
    tt = time.time()
    print "  Tuning"
    xdat = gaussians_tr
    #xdat =images_pca_tr[:,0:30]
    ydat =labels_tr
    #xtest =images_pca_te[:,0:30]
    xtest = gaussians_te

    @optunity.cross_validated(x=xdat, y=ydat, num_folds=10, num_iter=2)
    def svm_acc(x_train, y_train, x_test, y_test, C, gamma):
        model = svm.SVC(C=C, gamma=gamma,probability=True).fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return optunity.metrics.accuracy(y_test, y_pred)


    # 6. perform tuning
    optimal_pars, optimal_pars2, _ = optunity.maximize(svm_acc, num_evals=100, C=[0, 10], gamma=[0, 1])
    optimal_pars['probability']=True
    print "  Tuning: %0.2f sec" % optimal_pars2.stats["time"]

    # 7. train model on the full training set with tuned hyperparameters
    model =svm.SVC(**optimal_pars)
    optimal_model = model.fit(xdat, ydat)
    predict = optimal_model.predict(xtest)
    predict_prob = optimal_model.predict_proba(xtest)
    #print'  Optimal params C: %d, gamma: %d'%(round(optimal_pars["C"],3),round(optimal_pars["gamma"],3))
    print'  Optimal params C: %f, gamma: %f'%(optimal_pars["C"],optimal_pars["gamma"])
    # drawing
    drawimg(images_te,predict,dir_fig,(kind+"_89"),pixel)


    # 8. evaluation--
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    print confusion_matrix(labels_te, predict)

    # Accuracy rate
    from sklearn.metrics import accuracy_score
    print  "  Accuracy rate: %0.2f" % accuracy_score(labels_te, predict)

    #Classification Report--Precision、Recall, F-value
    from sklearn.metrics import classification_report
    print(classification_report(labels_te, predict, target_names=str(target_names)))

    # Compute Precision-Recall and plot curve
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc
    precision, recall, thresholds = precision_recall_curve(labels_tebi, map(round,predict_prob[:, 1]))
    area = auc(recall, precision)
    print "  Area Under the Precision-Recall Curve: %0.2f" % area

    # ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(labels_tebi, predict_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    print "  Area under the ROC curve: %f" % roc_auc



    print "\n\nfinish"



if __name__ == '__main__':
    main()

