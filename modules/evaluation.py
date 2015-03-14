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
import pylab as pl
from sklearn import svm, datasets

def roc(labels, prob, kind):
    from sklearn.utils import shuffle
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(labels, prob)
    roc_auc = auc(fpr, tpr)
    print "  Area under the ROC curve of %s : %0.4f" % (kind,roc_auc)

    """
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    #pl.xlim([0.0, 1.0])
    #pl.ylim([0.0, 1.0])
    pl.xlim([0.0, 0.5])
    pl.ylim([0.5, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()
    """

def pre_recall(labels, prob):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc
    #precision, recall, thresholds = precision_recall_curve(labels_tebi, map(round,prob))
    precision, recall, thresholds = precision_recall_curve(labels, prob)
    area = auc(recall, precision)
    print "  Area Under the Precision-Recall Curve of %s : %0.4f" % (kind,area)

