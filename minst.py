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

SHOW=1
DEBUG=0

dataSets = ['mnist_basic','mnist_background_images','mnist_background_random','mnist_rotation','mnist_rotation_back_image','mnist_rotation_new']
featureExtrs = ['PCA','ChainBlur','Gabor','Gabor_opt1']
#featureExtrs = ['Gabor']
opts = ['Default','CV','Gsearch','Optunity']
solvers = ['particle swarm','grid search','random search','cma-es','nelder-mead']
img_tr_all=lab_tr_all=img_te_all=lab_te_all=img_vl_all=lab_vl_all=[]
fparams=''  # feature extraction's params
bname = ''  # basic filename

if DEBUG==1:
    print 'DEBUG MODE'
    #featureExtrs = ['PCA']
    #featureExtrs = ['Gabor_opt1']
    #featureExtrs = ['ChainBlur']
    featureExtrs = ['Gabor','Gabor_cv']
    #opts = ['Default','Gsearch']

class Mnist():


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

    def appendBList(self,lists):
        for i in lists:
            if type(i)!=str:i=str(i)
            self.basicResults.append(i)

    def appendRList(self,orglist,lists):
        for i in lists:
            if type(i)!=str:i=str(i)
            orglist.append(i)
        return orglist

    def createCombs(self,numgs,numps,nfols,nitrs):
        combs=[]
        for numg in numgs:
            for nump in numps:
                for nfol in nfols:
                    for nitr in nitrs:
                        combs.append([numg,nump,nfol,nitr])
        return combs

    ##################################################
    ##################################################
    #def main(targetNumber,pixel,dataName,featureExtr,dir_output):
    def main(self,targetNumber,featureExtr,bname):

        #################################
        name = bname.split('_')[0]+'_'+bname.split('_')[2]
        # 1. Extract the specific number based on "target_names"
        target_names =([targetNumber],np.delete(np.linspace(0,9,10,np.int16),targetNumber))

        # class(0:target, 1:rest of all)
        labels_trbi,labels_vlbi,labels_tebi= [selectNum(dat,target_names)for dat in [self.lab_tr_all,self.lab_vl_all,self.lab_te_all]]

        # set binary data to Y
        self.ytrain =labels_trbi

        self.basicResults.extend([target_names[0][0],-target_names[0][0],len(labels_trbi[labels_trbi==0]),len(labels_trbi[labels_trbi==1]),len(labels_tebi[labels_tebi==0]),len(labels_tebi[labels_tebi==1])])
        print "  #train:%d (%d: %d, %d: %d)"%(len(labels_trbi),target_names[0][0],len(labels_trbi[labels_trbi==0]),-target_names[0][0],len(labels_trbi[labels_trbi==1]))
        print "  #test: %d (%d: %d, %d: %d)"%(len(labels_tebi),target_names[0][0],len(labels_tebi[labels_tebi==0]),-target_names[0][0],len(labels_tebi[labels_tebi==1]))
        print "  #val: %d (%d: %d, %d: %d)"%(len(labels_vlbi),target_names[0][0],len(labels_vlbi[labels_vlbi==0]),-target_names[0][0],len(labels_vlbi[labels_vlbi==1]))

        #################################
        # 2. Optimization
        ROCs = {}
        cvnum=10
        if np.min(cvnum<np.bincount(labels_vlbi))<cvnum:cvnum=np.min(np.bincount(labels_vlbi))

        for param in opts:
            print '\n   '+param
            tt=0
            t = time.time()
            results = copy.copy(self.basicResults)
            results.append(param)
            crange = np.logspace(-4,1.5,num=15,base=10.0)
            grange = np.logspace(-4,1.5,num=15,base=10.0)

            if param == 'Default':
                model =svm.SVC(kernel='rbf',probability=True)
                results.extend([model.C,model.gamma,0.0])

            elif param=='CV':
                av=0
                for c in crange:
                    for g in grange:
                        clf =svm.SVC(kernel='rbf',C=c,gamma=g,probability=True)
                        scores = cross_validation.cross_val_score(clf,self.xval,labels_vlbi,cv=cvnum, scoring='roc_auc')
                        if(av<np.mean(scores)):
                            av=np.mean(scores)
                            model=svm.SVC(kernel='rbf',C=c,gamma=g,probability=True)
                            tmp=[c,g]
                tt=time.time()-t
                tmp.append(tt)
                results.extend(tmp)

            elif param == 'Gsearch':
                args = {'kernel': ['rbf'], 'C':crange, 'gamma':grange}
                for score_name, score_func in [('roc_auc', roc_auc_score)]:
                    clf = GridSearchCV(svm.SVC(), args, scoring=score_name,cv=cvnum)
                    clf.fit(self.xval,labels_vlbi)
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

                args = {'num_evals':100, 'C':[min(crange),max(crange)], 'gamma':[min(grange),max(grange)]}
                optimal_pars, optimal_pars2, _ = optunity.maximize(svm_acc, **args)
                optimal_pars['probability']=True
                optimal_pars['kernel']='rbf'

                model =svm.SVC(**optimal_pars)
                tt = optimal_pars2.stats["time"]
                results.extend([optimal_pars["C"],optimal_pars["gamma"],tt])
            print '   %s: %0.2f sec'%(param,tt)
            print '    C:%0.2f, g:%0.2f'%(float(results[-3]),float(results[-2]))

            #################################
            # 3. Train model
            for data in ['test','train']:
                print '     '+data
                results.insert(14,data)
                if data == 'test':
                    xpredict,labels_true=self.xtest,labels_tebi
                elif data=='train':
                    xpredict,labels_true=self.xtrain,labels_trbi

                # fit to the test data
                optimal_model = model.fit(self.xtrain, self.ytrain)
                ypredict = optimal_model.predict(xpredict)
                ypredict_score = optimal_model.decision_function(xpredict)

                name = bname+data+'_'+param
                #################################
                # 4. Evaluation
                e=Evaluation(labels_true,ypredict,ypredict_score,self.dir_output,name,results,target_names)
                results,roc=e.evMain()
                if not ROCs.has_key(data): ROCs[data]={param:roc}
                else:ROCs[data].update({param:roc})

                # export
                #global fparams
                results.append(self.fparams)
                self.csvWriter.writerow(results)
                results = copy.copy(results)[0:15]
                results.remove(data)
        return ROCs

    ##################################################
    ##################################################
    def classification(self,dgts,featureExtr,tt):
        for i in range(dgts[0],dgts[1]):
            self.basicResults=[]
            f = open(self.csvname, 'ab+')
            self.csvWriter = csv.writer(f)

            bname = self.dataName+'_'+featureExtr+'_'+str(i)+'_'
            self.basicResults.extend([self.dataName,self.numEx,featureExtr,tt])
            dir_output_dgt =os.path.join(self.dir_output,str(i))
            if not os.path.exists(dir_output_dgt):os.mkdir(dir_output_dgt)

            print "\n%s, digit %d "%(self.dataName,i)
            rocs = self.main(i,featureExtr,bname)
            multipleROC(rocs,dir_output_dgt,bname)
            del self.csvWriter
            f.close()


##################################################
##################################################
if __name__ == '__main__':

    m=Mnist()
    # set Parameters
    m.pixel = 28 # pixel size (only for a square)
    m.dir_input = "..\data\input"
    m.dir_output = "..\data\output"
    m.dataSet =dataSets[0]
    m.numEx = int(time.time())
    m.trainSize=m.valSize=100
    m.teSize=100 #50000
    if DEBUG==1:m.trainSize=m.valSize=m.teSize=40
    m.normalization = False # MNIST is already normalized

    # file setting
    m.csvname=os.path.join(m.dir_output,'basicResults.csv')
    m.dir_output =os.path.join(m.dir_output,str(m.numEx))
    os.mkdir(m.dir_output)
    m.dataName = m.dataSet.split('_')[1]

    print 'No. %d starts'%m.numEx
    #################################
    # 1. downloade data
    #obtain(dir_input,dataSet)

    #################################
    # 2. make Varidation data
    #makeVarid(dir_input,dataSet)
    #checkLines(dir_input,dataSet)

    #################################
    # 3. data preprocessing
    img_tr_all,m.lab_tr_all,img_vl_all,m.lab_vl_all,img_te_all,m.lab_te_all \
        = load(m.dir_input,m.dataSet,m.pixel,trlen=m.trainSize,telen=m.teSize,vlen=m.valSize)
    #datas = load(dir_input,dataSet,pixel,trlen=trainSize,vlen=valSize,telen=teSize)
    #writeout(datas,dir_output)
    if m.normalization: img_tr_all,img_te_all,img_te_all = [norm(indat) for indat in [img_tr_all,img_te_all,img_te_all]]
    drawimg(img_tr_all,m.lab_tr_all,m.dir_output,"char_check_",m.pixel,basic=True)

    #################################
    # 4. Extract features & classification
    for featureExtr in featureExtrs:
        t = time.time()
        print '\n'+featureExtr
        bname = m.dataName+'_'+featureExtr+'_'

        ws_gb=[3,4,5,7]
        lms_gb=[1,1000]# just for opt1, will be devided by 100
        nbPhis_gb=[2,4,6,8]# the value should be more than 1
        ks_gb=[3,5,7,13]
        ns_gb=[2,4,7,14]
        sigs_gb=[1,50]# just for opt, will be devided by 10
        numgs_gb=[3,5]
        numps_gb=[4,10]
        nfols_gb=[2,4]
        nitrs_gb=[1,2]
        digits=[0,5]# digits number for classification

        if DEBUG ==1:
            ks_gb=[3,9,13]
            ws_gb=[3,5]
            nbPhis_gb=[2,6]#[4,6]
            ns_gb=[4,7,14,28]
            numgs_gb=[2]
            numps_gb=[2]
            nfols_gb=[2]
            nitrs_gb=[1]

        optparms=m.createCombs(numgs_gb,numps_gb,nfols_gb,nitrs_gb)
        m.fparams=[]

        # 4.1 PCA
        if(featureExtr=='PCA'):
            m.xtrain,m.xtest,m.xval,m.fparams = PCA().main([img_tr_all,img_te_all,img_vl_all],m.dir_output,bname)
            print "%s: %0.2f sec, CP:%d" %(featureExtr,(time.time()-t),m.fparams)
            m.classification(digits,featureExtr,time.time()-t)

        # 4.2. Chaincode/Bluring
        # directional decomposition of image into directional sub-images
        elif(featureExtr=='ChainBlur'):
            m.xtrain,m.xtest,m.xval=[chaincode(dat,m.pixel,m.dir_output)[0]for dat in[img_tr_all.copy(),img_te_all.copy(),img_vl_all.copy()]]
            print "%s: %0.2f sec" %(featureExtr,(time.time()-t))
            m.classification(digits,featureExtr,time.time()-t)

        # 4.3.1 Gabor
        elif(featureExtr=='Gabor'):
            gab= Gabor(m.pixel,img_tr_all,img_te_all,img_vl_all,\
                        m.dir_input,m.dir_output,m.dataName,\
                        ws=ws_gb,nbPhis=nbPhis_gb,ns=ns_gb)
            m.xtrain,m.xtest,m.xval,m.fparams=gab.gaborMain()
            if 'Gabor_cv' in featureExtrs:gab_cv= Gabor_cv(gab)
            print "%s: %0.2f sec" %(featureExtr,(time.time()-t))
            m.classification(digits,featureExtr,time.time()-t)


        # 4.3.1 Gabor
        elif(featureExtr=='Gabor_opt1'):
            for i in range(0,len(optparms)):
                t=time.time()
                gab= Gabor_opt1(m.pixel,img_tr_all,img_te_all,img_vl_all,\
                            m.dir_input,m.dir_output,m.dataName,\
                            ws=ws_gb,lms=lms_gb,nbPhis=nbPhis_gb,sigmas=sigs_gb,\
                            ksizes=ks_gb,ns=ns_gb,optparms=optparms[i])
                m.xtrain,m.xtest,m.xval,m.fparams=gab.gaborMain()
                print "%s: %0.2f sec" %(featureExtr,(time.time()-t))
                m.classification(digits,featureExtr,time.time()-t)

    print "finish"

