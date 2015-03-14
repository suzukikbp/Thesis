# encoding: utf-8

#-------------------------------------------------------------------------------
# Name:        pca
# Purpose:
#
# Author:      KSUZUKI
#
# Created:     04/02/2015
# Copyright:   (c) KSUZUKI 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import os,cv2,time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score


def pca(dat_tr,dat_te,target_names,pix):
    tt = time.time()
    dat = np.vstack([dat_tr,dat_te])
    print "  PCA"
    print "  org_image %s"%str(dat.shape)
    numCom =pix**2
    pca_scores = []
    for n in range(0,numCom):
        print n
        pca = PCA(n_components=n)
        pca.fit(dat)
        dat_pca= pca.transform(dat)
        if (sum(pca.explained_variance_ratio_)>0.8):
            pca_scores.append(np.mean(cross_val_score(pca, dat)))
            break

    print "  pca_image %s"%str(dat_pca.shape)
    print "  PCA: %.1f sec" %((time.time()-tt))

    return dat_pca[0:dat_tr.shape[0],],dat_pca[dat_tr.shape[0]:,],pca_scores

    """
    coeff=images_pca[:, 0]
    Ar = dot(coeff,score).T+mean(images)
    imshow(flipud(Ar))
    gray()
    """

def plotpca(target_names,images_pca,labels,kind,dir_path):
    colors = [plt.cm.spectral(i/10., 1) for i in range(10)]
    plt.figure()
    for c, target_name  in zip(colors, target_names):
        plt.scatter(images_pca[labels== target_name, 0], images_pca[labels== target_name, 1], c=c, label = str(target_name))
    plt.legend(loc=3)
    plt.xlabel('Factor 1')
    plt.ylabel('Factor 2')
    plt.title('PCA_'+kind)
    plt.savefig(os.path.join(dir_path,"pcaplot_"+kind+".png"), dpi=100)
    #plt.show()
